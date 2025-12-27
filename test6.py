"""
test6.py - 方法2: 入力画像の位置情報を保持したまま比較

処理フロー:
1. 入力画像のマスクを「中央配置せず」元の位置関係を保持したまま正規化
2. SQLiteデータベースから全パターンを読み込み
3. 各パターンに対して、入力画像の位置に合わせてオフセットして比較
4. 最適な(角度, dx, dy)を出力

特徴:
- 入力画像でうさぎが右にいれば、その位置情報を活用して探索
- 3Dシルエットを入力画像の重心位置へシフトして比較
"""

from pathlib import Path
import numpy as np
import cv2
import sqlite3
import trimesh
from PIL import Image, ImageDraw
from tqdm import tqdm


# ==========================
# 設定
# ==========================
BASE_DIR = Path(__file__).parent

INPUT_IMAGE = BASE_DIR / "Image0.png"
MODEL_PATH = BASE_DIR / "models_rabit_obj" / "rabit.obj"
DB_PATH = BASE_DIR / "model_features.db"

OUT_DIR = BASE_DIR / "rotation_results_test6"

# データベース内のシルエットサイズ
FEATURE_SIZE = (32, 32)

# 結果表示用の高解像度レンダリング
OUTPUT_SIZE = (512, 512)

# 結果保存数
TOP_K = 10

# 位置シフトの追加探索（粗い探索後の微調整）
FINE_SHIFT_RANGE = 50   # ピクセル
FINE_SHIFT_STEP = 5     # ピクセル

# ZYX回転系
INITIAL_RX = -90


# ==========================
# ユーティリティ
# ==========================
def imread_unicode(path: Path):
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def preprocess_image_to_mask(path: Path):
    """入力画像から物体マスクを抽出"""
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
        raise RuntimeError("物体が検出できない")

    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    return (labels == largest).astype(np.uint8)


def get_centroid(mask):
    """マスクの重心を計算"""
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    return float(xs.mean()), float(ys.mean())


def normalize_input_preserve_position(mask_full, size_wh):
    """
    入力マスクを正規化するが、画像内での相対位置を保持

    戻り値:
    - normalized_mask: 正規化されたマスク
    - position_info: 位置情報
    """
    w, h = int(size_wh[0]), int(size_wh[1])
    orig_h, orig_w = mask_full.shape[:2]

    ys, xs = np.where(mask_full > 0)
    if xs.size == 0:
        return np.zeros((h, w), dtype=np.uint8), None

    # 元画像でのバウンディングボックス
    rmin, rmax = int(ys.min()), int(ys.max())
    cmin, cmax = int(xs.min()), int(xs.max())

    # 元画像での重心（相対位置: 0.0〜1.0）
    orig_center_x_rel = float(xs.mean()) / orig_w
    orig_center_y_rel = float(ys.mean()) / orig_h

    # クロップしてスケーリング
    cropped = mask_full[rmin:rmax+1, cmin:cmax+1]
    height = rmax - rmin + 1
    width = cmax - cmin + 1

    # 80%スケールで正規化
    scale = min(w, h) * 0.8 / max(width, height) if max(width, height) > 0 else 1.0

    new_w = max(1, int(round(width * scale)))
    new_h = max(1, int(round(height * scale)))

    cropped_img = Image.fromarray((cropped * 255).astype(np.uint8))
    resized = cropped_img.resize((new_w, new_h), Image.NEAREST)
    resized_arr = (np.array(resized) > 127).astype(np.uint8)

    # 元の相対位置を保持して配置
    # 正規化後の画像での目標位置
    target_center_x = int(orig_center_x_rel * w)
    target_center_y = int(orig_center_y_rel * h)

    # 配置位置を計算
    x0 = target_center_x - new_w // 2
    y0 = target_center_y - new_h // 2

    # 画像内に収まるようにクリップ
    x0 = max(0, min(w - new_w, x0))
    y0 = max(0, min(h - new_h, y0))

    out = np.zeros((h, w), dtype=np.uint8)
    out[y0:y0+new_h, x0:x0+new_w] = resized_arr

    position_info = {
        'orig_center_x_rel': orig_center_x_rel,
        'orig_center_y_rel': orig_center_y_rel,
    }

    return out, position_info


def spherical_to_rotation(theta_deg, phi_deg, roll_deg=0.0):
    rx = INITIAL_RX + float(phi_deg)
    ry = -float(theta_deg)
    rz = float(roll_deg)
    return rx, ry, rz


def load_mesh_normalized(model_path: Path):
    """3Dモデルを正規化して読み込み"""
    mesh = trimesh.load(str(model_path), force="mesh")
    if isinstance(mesh, trimesh.Scene):
        meshes = [g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if not meshes:
            raise RuntimeError("Trimesh.Scene内にTrimeshが無い")
        mesh = trimesh.util.concatenate(meshes)

    mesh.vertices -= mesh.centroid
    ext = mesh.bounding_box.extents.max()
    if ext > 0:
        mesh.vertices /= ext
    return mesh


def render_silhouette(mesh, theta, phi, roll, size_wh):
    """シルエットをレンダリング（中央配置、80%スケール）- ZYX順"""
    w, h = int(size_wh[0]), int(size_wh[1])

    rx, ry, rz = spherical_to_rotation(theta, phi, roll)

    m = mesh.copy()
    Rz = trimesh.transformations.rotation_matrix(np.radians(rz), [0, 0, 1])
    Ry = trimesh.transformations.rotation_matrix(np.radians(ry), [0, 1, 0])
    Rx = trimesh.transformations.rotation_matrix(np.radians(rx), [1, 0, 0])

    # ZYX順で適用（test2.pyと同じ）
    m.apply_transform(Rz)
    m.apply_transform(Ry)
    m.apply_transform(Rx)

    verts = m.vertices[:, :2]

    min_xy = verts.min(axis=0)
    max_xy = verts.max(axis=0)
    extent = max_xy - min_xy
    scale = min(w, h) * 0.8 / max(extent[0], extent[1]) if max(extent) > 0 else 1.0

    center = (min_xy + max_xy) / 2
    proj_x = (verts[:, 0] - center[0]) * scale + w / 2
    proj_y = h / 2 - (verts[:, 1] - center[1]) * scale

    img = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(img)

    for f in m.faces:
        draw.polygon([(proj_x[i], proj_y[i]) for i in f], fill=255)

    return (np.array(img, dtype=np.uint8) > 127).astype(np.uint8)


def shift_mask(mask, dx, dy):
    """マスクを平行移動"""
    h, w = mask.shape[:2]
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    shifted = cv2.warpAffine(
        (mask * 255).astype(np.uint8),
        M,
        (w, h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )
    return (shifted > 127).astype(np.uint8)


def calculate_iou(mask_a, mask_b):
    """IoUを計算"""
    a = mask_a.astype(bool)
    b = mask_b.astype(bool)
    inter = np.logical_and(a, b).sum()
    uni = np.logical_or(a, b).sum()
    if uni == 0:
        return 0.0
    return float(inter) / float(uni)


def bytes_to_bits(data, size=(32, 32)):
    """BLOBデータを32x32のboolマスクに変換"""
    bits = []
    for byte in data:
        for i in range(8):
            bits.append((byte >> i) & 1)

    bits = bits[:size[0] * size[1]]
    return np.array(bits, dtype=np.uint8).reshape(size)


def calculate_centroid_offset(input_mask, model_mask):
    """
    入力マスクとモデルマスクの重心差からシフト量を計算

    - 入力マスク: 位置保持で正規化されている
    - モデルマスク: 中央配置
    - → モデルマスクを入力マスクの位置へシフト
    """
    # 入力マスクの実際の重心を計算
    input_centroid = get_centroid(input_mask)
    if input_centroid is None:
        return 0, 0

    # モデルマスクの重心を計算
    model_centroid = get_centroid(model_mask)
    if model_centroid is None:
        return 0, 0

    # シフト量: モデルを入力の位置へ移動
    dx = int(round(input_centroid[0] - model_centroid[0]))
    dy = int(round(input_centroid[1] - model_centroid[1]))

    return dx, dy


def save_overlay(input_mask, model_mask, out_path: Path):
    """オーバーレイ画像を保存（赤=入力、緑=モデル、黄=重なり）"""
    h, w = input_mask.shape[:2]
    overlay = np.zeros((h, w, 3), dtype=np.uint8)
    overlay[:, :, 2] = (input_mask * 255).astype(np.uint8)  # R
    overlay[:, :, 1] = (model_mask * 255).astype(np.uint8)  # G
    Image.fromarray(overlay).save(out_path)


# ==========================
# メイン
# ==========================
def main():
    OUT_DIR.mkdir(exist_ok=True)

    print("=" * 60)
    print("test6 - 方法2: 入力画像の位置情報を保持したまま比較")
    print("=" * 60)

    # 入力画像の読み込み（位置情報を保持）
    print("\n[1/5] 入力画像処理（位置情報保持）...")
    input_mask_full = preprocess_image_to_mask(INPUT_IMAGE)

    # 32x32に正規化（位置保持版）
    input_mask_32x32, position_info = normalize_input_preserve_position(input_mask_full, FEATURE_SIZE)

    if position_info:
        print(f"  入力: {INPUT_IMAGE.name}")
        print(f"  元画像での相対位置: x={position_info['orig_center_x_rel']:.3f}, y={position_info['orig_center_y_rel']:.3f}")

        # 32x32での実際の重心位置
        actual_centroid = get_centroid(input_mask_32x32)
        if actual_centroid:
            print(f"  32x32での重心位置: x={actual_centroid[0]:.1f}, y={actual_centroid[1]:.1f}")
            if actual_centroid[0] > 16:
                print(f"  → うさぎは画像の右側にいます")
            elif actual_centroid[0] < 16:
                print(f"  → うさぎは画像の左側にいます")
    else:
        raise RuntimeError("入力画像の処理に失敗")

    # データベース接続
    print("\n[2/5] データベース読み込み中...")
    if not DB_PATH.exists():
        raise FileNotFoundError(f"データベースが見つかりません: {DB_PATH}")

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM features")
    total_patterns = cursor.fetchone()[0]
    print(f"  データベース: {DB_PATH.name}")
    print(f"  事前計算パターン数: {total_patterns:,}")

    # 全パターンとマッチング（位置シフト付き）
    print("\n[3/5] 位置シフト付きマッチング中...")
    cursor.execute("SELECT id, theta, phi, roll, silhouette FROM features")

    results = []

    for row in tqdm(cursor.fetchall(), desc="  進捗", total=total_patterns):
        id_val, theta, phi, roll, silhouette_blob = row

        # BLOBをマスクに変換
        silhouette_32x32 = bytes_to_bits(silhouette_blob, FEATURE_SIZE)

        # 入力マスクの実際の重心位置に合わせてシフト
        dx, dy = calculate_centroid_offset(input_mask_32x32, silhouette_32x32)
        shifted_sil = shift_mask(silhouette_32x32, dx, dy)

        # IoU計算
        iou = calculate_iou(input_mask_32x32, shifted_sil)

        results.append({
            'id': id_val,
            'theta': theta,
            'phi': phi,
            'roll': roll,
            'dx': dx,
            'dy': dy,
            'iou': iou
        })

    conn.close()

    # スコアでソート
    results.sort(key=lambda x: x['iou'], reverse=True)

    best = results[0]
    print(f"\n  最良マッチ（粗探索）:")
    print(f"    Theta={best['theta']}°, Phi={best['phi']}°, Roll={best['roll']}°")
    print(f"    dx={best['dx']}, dy={best['dy']}")
    print(f"    IoU={best['iou']:.4f}")

    # 上位候補に対して微調整探索
    print(f"\n[4/5] 上位{TOP_K}件に対して微調整探索中...")
    print(f"  シフト範囲: ±{FINE_SHIFT_RANGE}px, ステップ: {FINE_SHIFT_STEP}px")

    mesh = load_mesh_normalized(MODEL_PATH)
    input_mask_highres, _ = normalize_input_preserve_position(input_mask_full, OUTPUT_SIZE)

    # 高解像度でのスケール比
    scale_factor = OUTPUT_SIZE[0] / FEATURE_SIZE[0]

    final_results = []
    shift_values = list(range(-FINE_SHIFT_RANGE, FINE_SHIFT_RANGE + 1, FINE_SHIFT_STEP))

    for r in tqdm(results[:TOP_K * 2], desc="  微調整"):
        # 高解像度でシルエットをレンダリング
        model_sil = render_silhouette(mesh, r['theta'], r['phi'], r['roll'], OUTPUT_SIZE)

        # ベースのシフト量（32x32からスケールアップ）
        base_dx = int(r['dx'] * scale_factor)
        base_dy = int(r['dy'] * scale_factor)

        best_iou = 0.0
        best_dx, best_dy = base_dx, base_dy

        # 微調整探索
        for ddx in shift_values:
            for ddy in shift_values:
                dx = base_dx + ddx
                dy = base_dy + ddy
                shifted = shift_mask(model_sil, dx, dy)
                iou = calculate_iou(input_mask_highres, shifted)
                if iou > best_iou:
                    best_iou = iou
                    best_dx, best_dy = dx, dy

        final_results.append({
            'theta': r['theta'],
            'phi': r['phi'],
            'roll': r['roll'],
            'dx': best_dx,
            'dy': best_dy,
            'iou': best_iou,
            'coarse_iou': r['iou']
        })

    # 最終ソート
    final_results.sort(key=lambda x: x['iou'], reverse=True)

    # 結果表示
    print("\n" + "=" * 60)
    print("結果（上位10件）:")
    print("=" * 60)
    for i, r in enumerate(final_results[:10], 1):
        print(f"  #{i}: theta={r['theta']:+7.1f}°, phi={r['phi']:+6.1f}°, roll={r['roll']:+6.1f}°")
        print(f"       dx={r['dx']:+4d}, dy={r['dy']:+4d}, IoU={r['iou']:.4f} (粗探索IoU={r['coarse_iou']:.4f})")

    # 結果保存
    print(f"\n[5/5] 結果画像を保存中...")
    for i, r in enumerate(final_results[:TOP_K], 1):
        model_sil = render_silhouette(mesh, r['theta'], r['phi'], r['roll'], OUTPUT_SIZE)
        model_sil_shifted = shift_mask(model_sil, r['dx'], r['dy'])

        out_name = (
            f"rank{i:02d}"
            f"_theta{r['theta']:+07.1f}"
            f"_phi{r['phi']:+06.1f}"
            f"_roll{r['roll']:+06.1f}"
            f"_dx{r['dx']:+05d}"
            f"_dy{r['dy']:+05d}"
            f"_iou{r['iou']:.4f}"
            f".png"
        )
        save_overlay(input_mask_highres, model_sil_shifted, OUT_DIR / out_name)

    print(f"\n結果: {OUT_DIR}")
    print("=" * 60)
    print("完了！")
    print("=" * 60)


if __name__ == "__main__":
    main()
