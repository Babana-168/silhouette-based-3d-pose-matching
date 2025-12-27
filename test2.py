"""
高速3Dモデル姿勢マッチング（SQLiteデータベース使用）
事前計算された40万パターンから最適な角度を数秒で検索
32x32でのマッチングのみ実行（高速版）
"""

import numpy as np
import cv2
import sqlite3
import trimesh
from pathlib import Path
from PIL import Image, ImageDraw
from tqdm import tqdm

# ==========================
# 設定
# ==========================
BASE_DIR = Path(__file__).parent

INPUT_IMAGE = BASE_DIR / "Image0.png"
MODEL_PATH = BASE_DIR / "models_rabit_obj" / "rabit.obj"
DB_PATH = BASE_DIR / "model_features.db"

# データベース内のシルエットサイズ（generate_features_sqlite.pyと一致）
FEATURE_SIZE = (32, 32)

# 結果表示用の高解像度レンダリング
OUTPUT_SIZE = (512, 512)

# 結果保存数
TOP_K = 10

OUT_DIR = BASE_DIR / "rotation_results"

# ==========================
# 画像読み込み（日本語パス対応）
# ==========================
def imread_unicode(path: Path):
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)

# ==========================
# 入力画像 → マスク
# ==========================
def preprocess_image_to_mask(path: Path):
    img = imread_unicode(path)
    if img is None:
        raise FileNotFoundError(path)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, bin_img = cv2.threshold(gray, 10, 1, cv2.THRESH_BINARY)

    kernel = np.ones((5, 5), np.uint8)
    bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, kernel)
    bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, kernel)

    num, labels, stats, _ = cv2.connectedComponentsWithStats(bin_img)
    if num <= 1:
        raise RuntimeError("物体が検出できません")

    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    return (labels == largest).astype(np.uint8)

def normalize_to_feature_size(mask):
    """マスクを32x32に正規化（データベースと同じサイズ）"""
    # クロップ
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return np.zeros(FEATURE_SIZE, dtype=np.uint8)

    mask = mask[ys.min():ys.max()+1, xs.min():xs.max()+1]

    # 32x32にリサイズ
    mask_img = Image.fromarray((mask * 255).astype(np.uint8))
    normalized = mask_img.resize(FEATURE_SIZE, Image.NEAREST)

    return (np.array(normalized) > 127).astype(np.uint8)

def normalize_input_to_size(mask, size):
    """入力画像を指定サイズに正規化（3Dレンダリングと同じスケーリング）"""
    # クロップ
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return np.zeros(size, dtype=np.uint8)

    rmin, rmax = ys.min(), ys.max()
    cmin, cmax = xs.min(), xs.max()

    cropped = mask[rmin:rmax+1, cmin:cmax+1]

    # バウンディングボックスのサイズ
    height = rmax - rmin + 1
    width = cmax - cmin + 1

    # 3Dモデルと同じスケーリング（画像の80%）
    extent_x = width
    extent_y = height
    scale = min(size[0], size[1]) * 0.8 / max(extent_x, extent_y)

    new_height = int(height * scale)
    new_width = int(width * scale)

    # リサイズ
    cropped_img = Image.fromarray((cropped * 255).astype(np.uint8))
    resized = cropped_img.resize((new_width, new_height), Image.NEAREST)
    resized_arr = (np.array(resized) > 127).astype(np.uint8)

    # 中央配置
    result = np.zeros(size, dtype=np.uint8)
    y_offset = (size[1] - new_height) // 2
    x_offset = (size[0] - new_width) // 2

    result[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized_arr

    return result

# ==========================
# IoU計算
# ==========================
def calculate_iou(mask1, mask2):
    """2つのマスクのIoU（Intersection over Union）を計算"""
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()

    if union == 0:
        return 0.0

    return intersection / union

# ==========================
# バイト→ビット変換
# ==========================
def bytes_to_bits(data, size=(32, 32)):
    """BLOBデータを32x32のboolマスクに変換"""
    bits = []
    for byte in data:
        for i in range(8):
            bits.append((byte >> i) & 1)

    bits = bits[:size[0] * size[1]]
    return np.array(bits, dtype=np.uint8).reshape(size)

# ==========================
# 回転とレンダリング
# ==========================
def spherical_to_rotation(theta_deg, phi_deg, roll_deg=0, initial_rx=-90):
    """データベースの角度を実際の回転角度に変換"""
    rx = initial_rx + phi_deg
    ry = -theta_deg
    rz = roll_deg
    return rx, ry, rz

def render_silhouette_at_size(mesh, theta, phi, roll, size):
    """任意のサイズでシルエットをレンダリング（theta, phi, rollは小数点可）"""
    rx, ry, rz = spherical_to_rotation(theta, phi, roll)

    m = mesh.copy()

    # ZYX順の回転を適用
    Rz = trimesh.transformations.rotation_matrix(np.radians(rz), [0, 0, 1])
    Ry = trimesh.transformations.rotation_matrix(np.radians(ry), [0, 1, 0])
    Rx = trimesh.transformations.rotation_matrix(np.radians(rx), [1, 0, 0])

    m.apply_transform(Rz)
    m.apply_transform(Ry)
    m.apply_transform(Rx)

    verts = m.vertices[:, :2]

    # スケーリングと中心配置
    min_xy = verts.min(axis=0)
    max_xy = verts.max(axis=0)
    extent = max_xy - min_xy
    scale = min(size[0], size[1]) * 0.8 / max(extent[0], extent[1]) if max(extent) > 0 else 1

    center = (min_xy + max_xy) / 2
    verts[:, 0] = (verts[:, 0] - center[0]) * scale + size[0] / 2
    verts[:, 1] = size[1] / 2 - (verts[:, 1] - center[1]) * scale

    img = Image.new("L", size, 0)
    draw = ImageDraw.Draw(img)

    for f in m.faces:
        draw.polygon([tuple(verts[i]) for i in f], fill=255)

    return np.array(img, dtype=np.uint8) > 127

# ==========================
# メイン処理
# ==========================
def main():
    print("=" * 60)
    print("高速3Dモデル姿勢マッチング（SQLite版・高速版）")
    print("=" * 60)

    # 結果フォルダをクリア
    import shutil
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir(exist_ok=True)
    print(f"\n結果フォルダを初期化: {OUT_DIR}")

    # 入力画像の読み込みと32x32正規化
    print("\n[1/4] 入力画像処理中...")
    input_mask_full = preprocess_image_to_mask(INPUT_IMAGE)
    input_mask_32x32 = normalize_to_feature_size(input_mask_full)
    print(f"  入力画像: {INPUT_IMAGE.name}")
    print(f"  マッチング用サイズ: {FEATURE_SIZE}")

    # データベース接続
    print("\n[2/4] データベース読み込み中...")
    if not DB_PATH.exists():
        raise FileNotFoundError(f"データベースが見つかりません: {DB_PATH}")

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # レコード数を確認
    cursor.execute("SELECT COUNT(*) FROM features")
    total_patterns = cursor.fetchone()[0]
    print(f"  データベース: {DB_PATH.name}")
    print(f"  事前計算パターン数: {total_patterns:,}")

    # 全パターンを読み込んでIoUを計算
    print("\n[3/4] 32x32で全パターンとマッチング中...")
    cursor.execute("SELECT id, theta, phi, roll, silhouette FROM features")

    results = []

    for row in tqdm(cursor.fetchall(), desc="  進捗", total=total_patterns):
        id_val, theta, phi, roll, silhouette_blob = row

        # BLOBをマスクに変換
        silhouette_32x32 = bytes_to_bits(silhouette_blob, FEATURE_SIZE)

        # IoU計算
        iou = calculate_iou(input_mask_32x32, silhouette_32x32)

        results.append({
            'id': id_val,
            'theta': theta,
            'phi': phi,
            'roll': roll,
            'iou': iou
        })

    conn.close()

    # スコアでソート（IoUが高い順）
    results.sort(key=lambda x: x['iou'], reverse=True)

    best = results[0]
    print(f"\n  最良マッチ:")
    print(f"    Theta={best['theta']}度, Phi={best['phi']}度, Roll={best['roll']}度")
    print(f"    IoU={best['iou']:.4f}")

    # 3Dモデルを読み込み
    print(f"\n[4/4] 結果保存中（上位{TOP_K}件）...")
    mesh = trimesh.load(MODEL_PATH, force="mesh")
    mesh.visual = None
    mesh.vertices -= mesh.centroid
    mesh.vertices /= mesh.bounding_box.extents.max()

    # 入力画像を可視化用に正規化
    input_mask_highres = normalize_input_to_size(input_mask_full, OUTPUT_SIZE)

    for i, result in enumerate(results[:TOP_K], 1):
        theta = result['theta']
        phi = result['phi']
        roll = result['roll']
        iou = result['iou']

        # 高解像度でレンダリング
        silhouette_highres = render_silhouette_at_size(mesh, theta, phi, roll, OUTPUT_SIZE)

        # オーバーレイ画像を作成（赤=入力、緑=3Dモデル、黄=重なり）
        overlay = np.zeros((OUTPUT_SIZE[1], OUTPUT_SIZE[0], 3), dtype=np.uint8)
        overlay[:, :, 0] = input_mask_highres * 255      # 赤チャンネル: 入力画像
        overlay[:, :, 1] = silhouette_highres * 255      # 緑チャンネル: 3Dシルエット

        # ファイル名に順位、角度、スコアを含める
        filename = f"rank{i:02d}_theta{theta:+07.1f}_phi{phi:+06.1f}_roll{roll:+06.1f}_iou{iou:.4f}.png"
        output_path = OUT_DIR / filename

        Image.fromarray(overlay).save(output_path)

        if i <= 5:
            print(f"  #{i}: Theta={theta:+6.1f}°, Phi={phi:+6.1f}°, Roll={roll:+6.1f}°, IoU={iou:.4f}")

    print(f"\n結果保存先: {OUT_DIR}")
    print("=" * 60)
    print("完了！")
    print("=" * 60)

if __name__ == "__main__":
    main()
