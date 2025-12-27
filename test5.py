"""
test5.py - 方法1: 上位候補に対して位置シフト付きマッチング

処理フロー:
1. test2の結果（rotation_results/rank*.png）から上位N件の角度を取得
2. 各角度候補に対して、dx,dyをグリッド探索してIoU最大を見つける
3. 最終的に最もIoUが高い(角度, dx, dy)の組み合わせを出力
"""

from pathlib import Path
import re
import numpy as np
import cv2
import trimesh
from PIL import Image, ImageDraw
from tqdm import tqdm


# ==========================
# 設定
# ==========================
BASE_DIR = Path(__file__).parent

INPUT_IMAGE = BASE_DIR / "Image0.png"
MODEL_PATH = BASE_DIR / "models_rabit_obj" / "rabit.obj"
TEST2_RESULT_DIR = BASE_DIR / "rotation_results"

OUT_DIR = BASE_DIR / "rotation_results_test5"

# 処理サイズ
RENDER_SIZE = (512, 512)

# 上位何件を探索するか
TOP_N_CANDIDATES = 20

# dx, dy の探索範囲（ピクセル）
SHIFT_RANGE = 150  # -150 ~ +150
SHIFT_STEP = 10    # 10ピクセル刻み

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


def normalize_input_to_render_size(mask_full, size_wh):
    """入力マスクをレンダリングサイズに正規化（3Dモデルと同じスケーリング）"""
    w, h = int(size_wh[0]), int(size_wh[1])

    ys, xs = np.where(mask_full > 0)
    if xs.size == 0:
        return np.zeros((h, w), dtype=np.uint8)

    rmin, rmax = int(ys.min()), int(ys.max())
    cmin, cmax = int(xs.min()), int(xs.max())
    cropped = mask_full[rmin:rmax+1, cmin:cmax+1]

    height = rmax - rmin + 1
    width = cmax - cmin + 1
    scale = min(w, h) * 0.8 / max(width, height) if max(width, height) > 0 else 1.0

    new_w = max(1, int(round(width * scale)))
    new_h = max(1, int(round(height * scale)))

    cropped_img = Image.fromarray((cropped * 255).astype(np.uint8))
    resized = cropped_img.resize((new_w, new_h), Image.NEAREST)
    resized_arr = (np.array(resized) > 127).astype(np.uint8)

    out = np.zeros((h, w), dtype=np.uint8)
    x0 = (w - new_w) // 2
    y0 = (h - new_h) // 2
    out[y0:y0+new_h, x0:x0+new_w] = resized_arr
    return out


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


def get_top_angles_from_test2(top_n):
    """test2の結果から上位N件の角度を取得"""
    if not TEST2_RESULT_DIR.exists():
        raise FileNotFoundError(f"test2の結果が無い: {TEST2_RESULT_DIR}")

    # rank*.pngを探す
    files = sorted(TEST2_RESULT_DIR.glob("rank*.png"))
    if not files:
        raise FileNotFoundError(f"rank*.pngが無い: {TEST2_RESULT_DIR}")

    results = []
    for f in files[:top_n]:
        filename = f.stem
        # rank01_theta+102.0_phi-090.0_roll+050.0_iou0.9138.png
        match = re.search(r'theta([+-]?\d+\.?\d*)_phi([+-]?\d+\.?\d*)_roll([+-]?\d+\.?\d*)_iou(\d+\.?\d*)', filename)
        if match:
            theta = float(match.group(1))
            phi = float(match.group(2))
            roll = float(match.group(3))
            iou = float(match.group(4))
            results.append({'theta': theta, 'phi': phi, 'roll': roll, 'iou': iou})

    return results


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
    print("test5 - 方法1: 上位候補に対して位置シフト付きマッチング")
    print("=" * 60)

    # test2の結果から上位N件の角度を取得
    print(f"\n[1/4] test2の結果を読み込み（上位{TOP_N_CANDIDATES}件）...")
    candidates = get_top_angles_from_test2(TOP_N_CANDIDATES)
    print(f"  取得した候補数: {len(candidates)}")

    for i, c in enumerate(candidates[:5], 1):
        print(f"    #{i}: theta={c['theta']:.1f}, phi={c['phi']:.1f}, roll={c['roll']:.1f}, iou={c['iou']:.4f}")

    # 入力画像を処理
    print("\n[2/4] 入力画像処理...")
    input_mask_full = preprocess_image_to_mask(INPUT_IMAGE)
    input_mask = normalize_input_to_render_size(input_mask_full, RENDER_SIZE)
    print(f"  入力: {INPUT_IMAGE.name}")

    # 3Dモデルを読み込み
    print("\n[3/4] 3Dモデル読み込み...")
    mesh = load_mesh_normalized(MODEL_PATH)
    print(f"  モデル: {MODEL_PATH.name}")

    # 各候補に対してdx,dyグリッド探索
    print(f"\n[4/4] 位置シフト付きマッチング中...")
    print(f"  シフト範囲: -{SHIFT_RANGE} ~ +{SHIFT_RANGE}, ステップ: {SHIFT_STEP}")

    shift_values = list(range(-SHIFT_RANGE, SHIFT_RANGE + 1, SHIFT_STEP))
    total_shifts = len(shift_values) ** 2
    print(f"  各候補あたり {total_shifts} パターン探索")

    all_results = []

    for cand_idx, cand in enumerate(tqdm(candidates, desc="  候補探索")):
        theta, phi, roll = cand['theta'], cand['phi'], cand['roll']

        # この角度でシルエットをレンダリング
        model_sil = render_silhouette(mesh, theta, phi, roll, RENDER_SIZE)

        best_iou = 0.0
        best_dx, best_dy = 0, 0

        # dx, dy グリッド探索
        for dx in shift_values:
            for dy in shift_values:
                shifted = shift_mask(model_sil, dx, dy)
                iou = calculate_iou(input_mask, shifted)
                if iou > best_iou:
                    best_iou = iou
                    best_dx, best_dy = dx, dy

        all_results.append({
            'theta': theta,
            'phi': phi,
            'roll': roll,
            'dx': best_dx,
            'dy': best_dy,
            'iou': best_iou,
            'original_iou': cand['iou']
        })

    # IoUでソート
    all_results.sort(key=lambda x: x['iou'], reverse=True)

    # 結果表示
    print("\n" + "=" * 60)
    print("結果（上位10件）:")
    print("=" * 60)
    for i, r in enumerate(all_results[:10], 1):
        print(f"  #{i}: theta={r['theta']:+7.1f}, phi={r['phi']:+6.1f}, roll={r['roll']:+6.1f}")
        print(f"       dx={r['dx']:+4d}, dy={r['dy']:+4d}, IoU={r['iou']:.4f} (元IoU={r['original_iou']:.4f})")

    # 上位結果を保存
    print(f"\n結果画像を保存中...")
    for i, r in enumerate(all_results[:10], 1):
        model_sil = render_silhouette(mesh, r['theta'], r['phi'], r['roll'], RENDER_SIZE)
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
        save_overlay(input_mask, model_sil_shifted, OUT_DIR / out_name)

    print(f"\n結果: {OUT_DIR}")
    print("=" * 60)
    print("完了！")
    print("=" * 60)


if __name__ == "__main__":
    main()
