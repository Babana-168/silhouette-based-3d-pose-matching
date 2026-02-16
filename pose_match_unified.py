# pose_match_unified.py
# 3Dポーズマッチング統合スクリプト
#
# 処理フロー:
# 1. SQLiteから特徴量検索で初期角度を取得
# 2. 粗い→細かいグリッドサーチで最適化
# 3. ワールドY座標クリッピング対応
# 4. マスク穴埋め対応
#
# 使用法: py -3.11 pose_match_unified.py
# 事前に generate_features_sqlite.py でDBを生成しておくこと

import math
import sqlite3
import time
from pathlib import Path
import numpy as np
import cv2
from PIL import Image, ImageDraw

# =============================================================================
# 設定
# =============================================================================

BASE_DIR = Path(__file__).parent
INPUT_IMAGE = BASE_DIR / "Image0.png"
MODEL_PATH = BASE_DIR / "models_rabit_obj" / "rabit_low.obj"
DB_PATH = BASE_DIR / "model_features.db"
OUT_DIR = BASE_DIR / "rotation_results_unified"

# レンダリング設定
CAMERA_FOV = 45.0
INITIAL_RX = -90.0
RENDER_SIZE = 512
NORM_SIZE = 256
BBOX_MARGIN_RATIO = 0.12
FEATURE_SIZE = (32, 32)

# 検索設定
TOP_N_CANDIDATES = 20  # 特徴量検索で上位何件を評価するか


# =============================================================================
# ユーティリティ関数
# =============================================================================

def imread_unicode(path):
    """Unicode対応画像読み込み"""
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def load_obj(path):
    """OBJファイル読み込み"""
    vertices, faces = [], []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            if parts[0] == "v" and len(parts) >= 4:
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif parts[0] == "f":
                idxs = []
                for p in parts[1:]:
                    idx = p.split("/")[0]
                    if idx:
                        idxs.append(int(idx) - 1)
                for i in range(1, len(idxs) - 1):
                    faces.append([idxs[0], idxs[i], idxs[i + 1]])
    return np.array(vertices, dtype=np.float32), np.array(faces, dtype=np.int32)


def normalize_vertices(v):
    """頂点を正規化"""
    v = v.copy()
    v -= v.mean(axis=0)
    v /= np.abs(v).max() * 2
    return v


def rot_x(deg):
    a = math.radians(deg)
    return np.array([[1,0,0],[0,math.cos(a),-math.sin(a)],[0,math.sin(a),math.cos(a)]], dtype=np.float32)


def rot_y(deg):
    a = math.radians(deg)
    return np.array([[math.cos(a),0,math.sin(a)],[0,1,0],[-math.sin(a),0,math.cos(a)]], dtype=np.float32)


def rot_z(deg):
    a = math.radians(deg)
    return np.array([[math.cos(a),-math.sin(a),0],[math.sin(a),math.cos(a),0],[0,0,1]], dtype=np.float32)


def bbox_from_mask(mask):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def expand_bbox(x0, y0, x1, y1, w, h, margin_ratio):
    bw, bh = x1 - x0, y1 - y0
    m = int(max(bw, bh) * margin_ratio)
    return max(0, x0 - m), max(0, y0 - m), min(w - 1, x1 + m), min(h - 1, y1 + m)


def crop_and_resize(img, bbox, size):
    x0, y0, x1, y1 = bbox
    crop = img[y0:y1+1, x0:x1+1]
    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_NEAREST)


def fill_holes(mask):
    """マスクの穴を埋める"""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(mask)
    cv2.drawContours(filled, contours, -1, 1, -1)
    return filled


# =============================================================================
# 特徴量抽出（SQLite検索用）
# =============================================================================

def compute_hu_moments(mask):
    """Huモーメントを計算"""
    mask_float = mask.astype(np.float64)
    y, x = np.mgrid[:mask.shape[0], :mask.shape[1]]
    m00 = mask_float.sum()

    if m00 == 0:
        return [0.0] * 4

    m10 = (x * mask_float).sum()
    m01 = (y * mask_float).sum()
    cx = m10 / m00
    cy = m01 / m00

    xc = x - cx
    yc = y - cy

    mu20 = (xc**2 * mask_float).sum() / m00
    mu02 = (yc**2 * mask_float).sum() / m00
    mu11 = (xc * yc * mask_float).sum() / m00
    mu30 = (xc**3 * mask_float).sum() / m00
    mu03 = (yc**3 * mask_float).sum() / m00
    mu21 = (xc**2 * yc * mask_float).sum() / m00
    mu12 = (xc * yc**2 * mask_float).sum() / m00

    hu1 = mu20 + mu02
    hu2 = (mu20 - mu02)**2 + 4*mu11**2
    hu3 = (mu30 - 3*mu12)**2 + (3*mu21 - mu03)**2
    hu4 = (mu30 + mu12)**2 + (mu21 + mu03)**2

    hu_log = []
    for h in [hu1, hu2, hu3, hu4]:
        if h != 0:
            hu_log.append(float(-np.sign(h) * np.log10(abs(h) + 1e-10)))
        else:
            hu_log.append(0.0)

    return hu_log


def extract_features_from_mask(mask):
    """マスクから特徴量を抽出"""
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None

    rmin, rmax = ys.min(), ys.max()
    cmin, cmax = xs.min(), xs.max()

    height = rmax - rmin
    width = cmax - cmin

    if height <= 0 or width <= 0:
        return None

    aspect_ratio = width / height
    area = mask.sum()
    bbox_area = height * width
    fill_ratio = area / bbox_area if bbox_area > 0 else 0

    cy = np.mean(ys)
    cx = np.mean(xs)
    rel_cy = (cy - rmin) / height
    rel_cx = (cx - cmin) / width

    hu_moments = compute_hu_moments(mask)

    # 正規化シルエット
    cropped = mask[rmin:rmax+1, cmin:cmax+1]
    cropped_img = Image.fromarray((cropped * 255).astype(np.uint8))
    normalized = cropped_img.resize(FEATURE_SIZE, Image.NEAREST)
    normalized_arr = np.array(normalized) > 127

    return {
        "aspect_ratio": aspect_ratio,
        "fill_ratio": fill_ratio,
        "centroid_x": rel_cx,
        "centroid_y": rel_cy,
        "hu_moments": hu_moments,
        "silhouette": normalized_arr
    }


def bytes_to_bits(data):
    """バイナリを1024個のboolに変換"""
    bits = []
    for byte in data:
        for i in range(8):
            bits.append(bool(byte & (1 << i)))
    return np.array(bits).reshape(FEATURE_SIZE)


def calculate_feature_distance(feat1, feat2_row):
    """特徴量間の距離を計算"""
    # feat2_rowはSQLiteから取得した行
    aspect_diff = abs(feat1["aspect_ratio"] - feat2_row["aspect_ratio"])
    fill_diff = abs(feat1["fill_ratio"] - feat2_row["fill_ratio"])

    centroid_diff = math.sqrt(
        (feat1["centroid_x"] - feat2_row["centroid_x"])**2 +
        (feat1["centroid_y"] - feat2_row["centroid_y"])**2
    )

    hu_diff = math.sqrt(
        (feat1["hu_moments"][0] - feat2_row["hu1"])**2 +
        (feat1["hu_moments"][1] - feat2_row["hu2"])**2 +
        (feat1["hu_moments"][2] - feat2_row["hu3"])**2 +
        (feat1["hu_moments"][3] - feat2_row["hu4"])**2
    )

    # シルエットIoU
    sil1 = feat1["silhouette"]
    sil2 = bytes_to_bits(feat2_row["silhouette"])
    intersection = np.logical_and(sil1, sil2).sum()
    union = np.logical_or(sil1, sil2).sum()
    sil_iou = intersection / union if union > 0 else 0
    sil_diff = 1.0 - sil_iou

    # 重み付け合計
    distance = (
        aspect_diff * 2.0 +
        fill_diff * 1.0 +
        centroid_diff * 1.0 +
        hu_diff * 0.5 +
        sil_diff * 3.0
    )

    return distance


def search_sqlite_features(target_features, db_path, top_n=TOP_N_CANDIDATES):
    """SQLiteから最も似た特徴を持つ角度を検索"""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # アスペクト比で絞り込み
    aspect = target_features["aspect_ratio"]
    cursor.execute('''
        SELECT * FROM features
        WHERE aspect_ratio BETWEEN ? AND ?
    ''', (aspect - 0.5, aspect + 0.5))

    rows = cursor.fetchall()
    conn.close()

    if not rows:
        return []

    # 距離計算
    distances = []
    for row in rows:
        dist = calculate_feature_distance(target_features, row)
        distances.append({
            "theta": row["theta"],
            "phi": row["phi"],
            "roll": row["roll"],
            "distance": dist
        })

    # ソート
    distances.sort(key=lambda x: x["distance"])

    return distances[:top_n]


# =============================================================================
# レンダリング
# =============================================================================

def render_silhouette_world_clip(v, faces, theta, phi, roll, cam_x, cam_y, cam_z, size, clip_world_y=None):
    """シルエットレンダリング（ワールドY座標クリッピング対応）"""
    rx = INITIAL_RX + phi
    ry = -theta
    rz = roll
    R = rot_x(rx) @ rot_y(ry) @ rot_z(rz)
    verts = (R @ v.T).T

    vx = verts[:, 0] - cam_x
    vy = verts[:, 1] - cam_y
    vz = cam_z - verts[:, 2]
    vz = np.clip(vz, 0.1, None)

    f = size / (2.0 * math.tan(math.radians(CAMERA_FOV / 2.0)))
    px = (vx / vz) * f + size * 0.5
    py = size * 0.5 - (vy / vz) * f

    sil = np.zeros((size, size), dtype=np.uint8)

    for a, b, c in faces:
        if clip_world_y is not None:
            if verts[a, 1] < clip_world_y or verts[b, 1] < clip_world_y or verts[c, 1] < clip_world_y:
                continue
        pts = np.array([[px[a], py[a]], [px[b], py[b]], [px[c], py[c]]], dtype=np.int32)
        cv2.fillConvexPoly(sil, pts.reshape(-1, 1, 2), 255)

    return (sil > 0).astype(np.uint8)


# =============================================================================
# IoU計算
# =============================================================================

def calc_iou(a, b):
    a, b = a.astype(bool), b.astype(bool)
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return inter / union if union > 0 else 0.0


# =============================================================================
# 最適化
# =============================================================================

def compute_iou_for_params(v_np, faces_np, target_norm, theta, phi, roll, cam_x, cam_y, cam_z, clip_y):
    """パラメータに対するIoUを計算"""
    sil = render_silhouette_world_clip(v_np, faces_np, theta, phi, roll, cam_x, cam_y, cam_z, RENDER_SIZE, clip_world_y=clip_y)
    bb = bbox_from_mask(sil)
    if bb is None:
        return 0.0
    x0, y0, x1, y1 = expand_bbox(*bb, RENDER_SIZE, RENDER_SIZE, BBOX_MARGIN_RATIO)
    sil_norm = crop_and_resize(sil, (x0, y0, x1, y1), NORM_SIZE)
    return calc_iou(target_norm, sil_norm)


def optimize_parameters(v_np, faces_np, target_norm, init_theta, init_phi, init_roll):
    """多段階グリッドサーチ最適化"""

    best_theta = init_theta
    best_phi = init_phi
    best_roll = init_roll
    best_cam_x = -0.45
    best_cam_y = -0.45
    best_cam_z = 2.72
    best_clip_y = -0.39

    best_iou = compute_iou_for_params(v_np, faces_np, target_norm,
                                       best_theta, best_phi, best_roll,
                                       best_cam_x, best_cam_y, best_cam_z, best_clip_y)
    print(f"[Initial] IoU: {best_iou*100:.4f}%")

    # Phase 1: 回転パラメータ（粗い探索）
    print("\n[Phase 1] Coarse rotation search...")
    step = 1.0
    for d_theta in np.arange(-5, 5.01, step):
        for d_phi in np.arange(-5, 5.01, step):
            for d_roll in np.arange(-5, 5.01, step):
                theta = init_theta + d_theta
                phi = init_phi + d_phi
                roll = init_roll + d_roll

                iou = compute_iou_for_params(v_np, faces_np, target_norm,
                                              theta, phi, roll,
                                              best_cam_x, best_cam_y, best_cam_z, best_clip_y)
                if iou > best_iou:
                    best_iou = iou
                    best_theta, best_phi, best_roll = theta, phi, roll
                    print(f"  theta={theta:.1f}, phi={phi:.1f}, roll={roll:.1f} -> IoU={iou*100:.4f}%")

    print(f"[Phase 1] Best: {best_iou*100:.4f}%")

    # Phase 2: カメラ位置
    print("\n[Phase 2] Camera position search...")
    for d_x in np.arange(-0.1, 0.101, 0.02):
        for d_y in np.arange(-0.1, 0.101, 0.02):
            for d_z in np.arange(-0.2, 0.201, 0.05):
                cam_x = -0.45 + d_x
                cam_y = -0.45 + d_y
                cam_z = 2.72 + d_z

                iou = compute_iou_for_params(v_np, faces_np, target_norm,
                                              best_theta, best_phi, best_roll,
                                              cam_x, cam_y, cam_z, best_clip_y)
                if iou > best_iou:
                    best_iou = iou
                    best_cam_x, best_cam_y, best_cam_z = cam_x, cam_y, cam_z
                    print(f"  cam=({cam_x:.3f}, {cam_y:.3f}, {cam_z:.3f}) -> IoU={iou*100:.4f}%")

    print(f"[Phase 2] Best: {best_iou*100:.4f}%")

    # Phase 3: 回転パラメータ（細かい探索）
    print("\n[Phase 3] Fine rotation search...")
    step = 0.2
    center_theta, center_phi, center_roll = best_theta, best_phi, best_roll
    for d_theta in np.arange(-1, 1.01, step):
        for d_phi in np.arange(-1, 1.01, step):
            for d_roll in np.arange(-1, 1.01, step):
                theta = center_theta + d_theta
                phi = center_phi + d_phi
                roll = center_roll + d_roll

                iou = compute_iou_for_params(v_np, faces_np, target_norm,
                                              theta, phi, roll,
                                              best_cam_x, best_cam_y, best_cam_z, best_clip_y)
                if iou > best_iou:
                    best_iou = iou
                    best_theta, best_phi, best_roll = theta, phi, roll
                    print(f"  theta={theta:.2f}, phi={phi:.2f}, roll={roll:.2f} -> IoU={iou*100:.4f}%")

    print(f"[Phase 3] Best: {best_iou*100:.4f}%")

    # Phase 4: クリップY位置
    print("\n[Phase 4] Clip Y search...")
    for clip_y in np.arange(-0.5, 0.01, 0.02):
        iou = compute_iou_for_params(v_np, faces_np, target_norm,
                                      best_theta, best_phi, best_roll,
                                      best_cam_x, best_cam_y, best_cam_z, clip_y)
        if iou > best_iou:
            best_iou = iou
            best_clip_y = clip_y
            print(f"  clip_y={clip_y:.3f} -> IoU={iou*100:.4f}%")

    print(f"[Phase 4] Best: {best_iou*100:.4f}%")

    return {
        "theta": best_theta,
        "phi": best_phi,
        "roll": best_roll,
        "cam_x": best_cam_x,
        "cam_y": best_cam_y,
        "cam_z": best_cam_z,
        "clip_y": best_clip_y,
        "iou": best_iou
    }


# =============================================================================
# 結果保存
# =============================================================================

def save_results(out_dir, target_norm, sil_norm, params, search_time, optimize_time):
    """結果を保存"""
    out_dir.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(out_dir / "target_norm.png"), target_norm * 255)
    cv2.imwrite(str(out_dir / "sil_best.png"), sil_norm * 255)

    # オーバーレイ
    overlay = np.zeros((NORM_SIZE, NORM_SIZE, 3), dtype=np.uint8)
    overlay[:, :, 2] = target_norm * 255
    overlay[:, :, 1] = sil_norm * 255
    cv2.imwrite(str(out_dir / "overlay.png"), overlay)

    # ミスマッチ
    t_bool = target_norm > 0
    m_bool = sil_norm > 0
    mismatch = np.zeros((NORM_SIZE, NORM_SIZE, 3), dtype=np.uint8)
    mismatch[np.logical_and(t_bool, ~m_bool), 2] = 255
    mismatch[np.logical_and(m_bool, ~t_bool), 1] = 255
    cv2.imwrite(str(out_dir / "mismatch.png"), mismatch)

    # 結果テキスト
    with open(out_dir / "result.txt", "w", encoding="utf-8") as f:
        f.write("=== Pose Match Unified Result ===\n\n")
        f.write("[Best Parameters]\n")
        f.write(f"  theta: {params['theta']:.4f}\n")
        f.write(f"  phi: {params['phi']:.4f}\n")
        f.write(f"  roll: {params['roll']:.4f}\n")
        f.write(f"  cam_x: {params['cam_x']:.4f}\n")
        f.write(f"  cam_y: {params['cam_y']:.4f}\n")
        f.write(f"  cam_z: {params['cam_z']:.4f}\n")
        f.write(f"  clip_y: {params['clip_y']:.4f}\n\n")
        f.write(f"[Result]\n")
        f.write(f"  IoU: {params['iou']*100:.4f}%\n\n")
        f.write(f"[Time]\n")
        f.write(f"  Feature search: {search_time:.2f}s\n")
        f.write(f"  Optimization: {optimize_time:.2f}s\n")
        f.write(f"  Total: {search_time + optimize_time:.2f}s\n")

    print(f"\nOutput: {out_dir}")


# =============================================================================
# メイン
# =============================================================================

def main():
    print("=" * 60)
    print("Pose Match Unified")
    print("=" * 60)

    # DBチェック
    if not DB_PATH.exists():
        print(f"\nError: {DB_PATH} not found.")
        print("Run generate_features_sqlite.py first.")
        return

    # 画像読み込み
    print("\n[1/5] Loading image...")
    original = imread_unicode(INPUT_IMAGE)
    H, W = original.shape[:2]
    print(f"  Size: {W}x{H}")

    # マスク作成（穴埋め）
    print("\n[2/5] Creating mask...")
    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 10, 1, cv2.THRESH_BINARY)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    if num > 1:
        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        mask = (labels == largest).astype(np.uint8)
    mask_filled = fill_holes(mask)
    print(f"  Mask created (holes filled)")

    # ターゲット正規化
    bb_target = bbox_from_mask(mask_filled)
    x0t, y0t, x1t, y1t = expand_bbox(*bb_target, W, H, BBOX_MARGIN_RATIO)
    target_norm = crop_and_resize(mask_filled, (x0t, y0t, x1t, y1t), NORM_SIZE)

    # 特徴量抽出
    print("\n[3/5] Extracting features...")
    target_features = extract_features_from_mask(mask_filled)
    if target_features is None:
        print("  Error: Feature extraction failed.")
        return
    print(f"  Aspect ratio: {target_features['aspect_ratio']:.3f}")
    print(f"  Fill ratio: {target_features['fill_ratio']:.3f}")

    # SQLite検索
    print(f"\n[4/5] Searching features (top {TOP_N_CANDIDATES})...")
    search_start = time.time()
    candidates = search_sqlite_features(target_features, DB_PATH, TOP_N_CANDIDATES)
    search_time = time.time() - search_start

    if not candidates:
        print("  Error: No candidates found.")
        return

    print(f"  Found {len(candidates)} candidates in {search_time:.2f}s")
    print(f"\n  Top 5:")
    for i, c in enumerate(candidates[:5]):
        print(f"    {i+1}. theta={c['theta']:+4d}, phi={c['phi']:+3d}, roll={c['roll']:+3d} (dist={c['distance']:.3f})")

    # 3Dモデル読み込み
    print("\n[5/5] Loading 3D model and optimizing...")
    v_np, faces_np = load_obj(MODEL_PATH)
    v_np = normalize_vertices(v_np)
    print(f"  Mesh: {len(v_np)} vertices, {len(faces_np)} faces")

    # 上位候補でIoU評価
    print("\n  Evaluating candidates...")
    best_candidate = None
    best_init_iou = 0.0

    for c in candidates:
        iou = compute_iou_for_params(v_np, faces_np, target_norm,
                                      c["theta"], c["phi"], c["roll"],
                                      -0.45, -0.45, 2.72, -0.39)
        if iou > best_init_iou:
            best_init_iou = iou
            best_candidate = c

    print(f"  Best candidate: theta={best_candidate['theta']}, phi={best_candidate['phi']}, roll={best_candidate['roll']}")
    print(f"  Initial IoU: {best_init_iou*100:.4f}%")

    # 最適化
    print("\n" + "=" * 60)
    print("Optimization")
    print("=" * 60)
    optimize_start = time.time()
    params = optimize_parameters(v_np, faces_np, target_norm,
                                  best_candidate["theta"],
                                  best_candidate["phi"],
                                  best_candidate["roll"])
    optimize_time = time.time() - optimize_start

    # 最終レンダリング
    sil = render_silhouette_world_clip(v_np, faces_np,
                                        params["theta"], params["phi"], params["roll"],
                                        params["cam_x"], params["cam_y"], params["cam_z"],
                                        RENDER_SIZE, clip_world_y=params["clip_y"])
    bb_sil = bbox_from_mask(sil)
    x0s, y0s, x1s, y1s = expand_bbox(*bb_sil, RENDER_SIZE, RENDER_SIZE, BBOX_MARGIN_RATIO)
    sil_norm = crop_and_resize(sil, (x0s, y0s, x1s, y1s), NORM_SIZE)

    # 結果保存
    print("\n" + "=" * 60)
    print("Saving Results")
    print("=" * 60)
    save_results(OUT_DIR, target_norm, sil_norm, params, search_time, optimize_time)

    print("\n" + "=" * 60)
    print(f"Final IoU: {params['iou']*100:.4f}%")
    print(f"Total time: {search_time + optimize_time:.2f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
