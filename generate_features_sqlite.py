"""
3Dモデル(OBJ)の全角度パターンの特徴をSQLiteに直接保存

JSONを経由せず直接SQLiteに書き込むことでメモリ使用量を削減
"""

import numpy as np
from PIL import Image, ImageDraw
import trimesh
from pathlib import Path
import warnings
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing
import threading

warnings.filterwarnings('ignore')

# スレッド数
NUM_THREADS = multiprocessing.cpu_count()

# ===== 設定 =====
RENDER_SIZE = (128, 128)
FEATURE_SIZE = (32, 32)
TARGET_FACES = 15000
INITIAL_RX = -90

# 角度の範囲
THETA_RANGE = range(-180, 180, 2)   # 水平方向 180パターン
PHI_RANGE = range(-90, 91, 2)       # 垂直方向 91パターン
ROLL_RANGE = range(-60, 61, 5)      # ロール 25パターン
# 合計: 180 * 91 * 25 = 409,500 パターン

# DB書き込み用ロック
db_lock = threading.Lock()


def create_rotation_matrix(rx_deg, ry_deg, rz_deg):
    """ZYX順序の回転行列を作成"""
    rx = np.radians(rx_deg)
    ry = np.radians(ry_deg)
    rz = np.radians(rz_deg)

    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)

    R = np.array([
        [cy * cz, -cy * sz, sy],
        [cx * sz + cz * sx * sy, cx * cz - sx * sy * sz, -cy * sx],
        [sx * sz - cx * cz * sy, cz * sx + cx * sy * sz, cx * cy]
    ], dtype=np.float32)

    return R


def spherical_to_rotation(theta_deg, phi_deg, roll_deg=0):
    rx = INITIAL_RX + phi_deg
    ry = -theta_deg
    rz = roll_deg
    return rx, ry, rz


def render_silhouette(vertices, faces, rx, ry, rz, size):
    """シルエットをレンダリング"""
    width, height = size
    R = create_rotation_matrix(rx, ry, rz)
    rotated = vertices @ R.T

    min_xy = rotated[:, :2].min(axis=0)
    max_xy = rotated[:, :2].max(axis=0)

    extent = max_xy - min_xy
    scale = min(width, height) * 0.8 / max(extent[0], extent[1]) if max(extent) > 0 else 1

    center = (min_xy + max_xy) / 2
    proj_x = (rotated[:, 0] - center[0]) * scale + width / 2
    proj_y = height / 2 - (rotated[:, 1] - center[1]) * scale

    img = Image.new('L', (width, height), 0)
    draw = ImageDraw.Draw(img)

    for face in faces:
        polygon = [(proj_x[face[0]], proj_y[face[0]]),
                   (proj_x[face[1]], proj_y[face[1]]),
                   (proj_x[face[2]], proj_y[face[2]])]
        draw.polygon(polygon, fill=255)

    return np.array(img, dtype=np.uint8) > 127


def get_bounding_box(mask):
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not np.any(rows) or not np.any(cols):
        return 0, 0, 0, 0
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return rmin, rmax, cmin, cmax


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


def bits_to_bytes(bits):
    """1024個のboolをバイナリに変換"""
    result = bytearray(128)
    for i, bit in enumerate(bits):
        if bit:
            result[i // 8] |= (1 << (i % 8))
    return bytes(result)


def extract_features(mask):
    """シルエットから特徴を抽出"""
    rmin, rmax, cmin, cmax = get_bounding_box(mask)

    if rmax <= rmin or cmax <= cmin:
        return None

    height = rmax - rmin
    width = cmax - cmin

    aspect_ratio = width / height if height > 0 else 1.0
    area = mask.sum()
    bbox_area = height * width
    fill_ratio = area / bbox_area if bbox_area > 0 else 0

    coords = np.where(mask)
    if len(coords[0]) == 0:
        rel_cy, rel_cx = 0.5, 0.5
    else:
        cy = np.mean(coords[0])
        cx = np.mean(coords[1])
        rel_cy = (cy - rmin) / height if height > 0 else 0.5
        rel_cx = (cx - cmin) / width if width > 0 else 0.5

    hu_moments = compute_hu_moments(mask)

    cropped = mask[rmin:rmax+1, cmin:cmax+1]
    cropped_img = Image.fromarray((cropped * 255).astype(np.uint8))
    normalized = cropped_img.resize(FEATURE_SIZE, Image.NEAREST)
    normalized_arr = np.array(normalized) > 127
    silhouette_bytes = bits_to_bytes(normalized_arr.flatten().tolist())

    return {
        "aspect_ratio": round(aspect_ratio, 4),
        "fill_ratio": round(fill_ratio, 4),
        "centroid_x": round(rel_cx, 4),
        "centroid_y": round(rel_cy, 4),
        "hu_moments": [round(h, 4) for h in hu_moments],
        "silhouette": silhouette_bytes
    }


def load_3d_model(model_path, target_faces=TARGET_FACES):
    mesh = trimesh.load(model_path, force='mesh')

    if isinstance(mesh, trimesh.Scene):
        meshes = [g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if meshes:
            mesh = trimesh.util.concatenate(meshes)

    mesh.vertices -= mesh.centroid
    scale = 1.0 / mesh.bounding_box.extents.max()
    mesh.vertices *= scale

    if len(mesh.faces) > target_faces:
        np.random.seed(42)
        indices = np.random.choice(len(mesh.faces), target_faces, replace=False)
        selected_faces = mesh.faces[indices]
        unique_vertices = np.unique(selected_faces.flatten())
        vertex_map = {old: new for new, old in enumerate(unique_vertices)}
        new_vertices = mesh.vertices[unique_vertices]
        new_faces = np.array([[vertex_map[v] for v in face] for face in selected_faces])
        mesh = trimesh.Trimesh(vertices=new_vertices, faces=new_faces)

    return mesh


def process_angle(args):
    """1つの角度パターンを処理"""
    theta, phi, roll, vertices, faces = args
    rx, ry, rz = spherical_to_rotation(theta, phi, roll)
    silhouette = render_silhouette(vertices, faces, rx, ry, rz, RENDER_SIZE)
    features = extract_features(silhouette)

    if features is not None:
        features["theta"] = theta
        features["phi"] = phi
        features["roll"] = roll
        return features
    return None


def create_database(db_path):
    """SQLiteデータベースを作成"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS metadata (
            key TEXT PRIMARY KEY,
            value TEXT
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS features (
            id INTEGER PRIMARY KEY,
            theta INTEGER NOT NULL,
            phi INTEGER NOT NULL,
            roll INTEGER NOT NULL,
            aspect_ratio REAL NOT NULL,
            fill_ratio REAL NOT NULL,
            centroid_x REAL NOT NULL,
            centroid_y REAL NOT NULL,
            hu1 REAL NOT NULL,
            hu2 REAL NOT NULL,
            hu3 REAL NOT NULL,
            hu4 REAL NOT NULL,
            silhouette BLOB NOT NULL
        )
    ''')

    conn.commit()
    return conn


def main():
    BASE_PATH = Path(__file__).resolve().parent
    MODEL_PATH = BASE_PATH / "models_rabit_obj" / "rabit.obj"
    DB_PATH = BASE_PATH / "model_features.db"

    print("=" * 60)
    print("  3Dモデル特徴量SQLite生成")
    print("=" * 60)
    print(f"  モデル: {MODEL_PATH}")
    print(f"  出力: {DB_PATH}")
    print(f"  スレッド数: {NUM_THREADS}")

    # 既存DBを削除
    if DB_PATH.exists():
        DB_PATH.unlink()

    # モデル読み込み
    print("\n[1/4] モデル読み込み...")
    mesh = load_3d_model(MODEL_PATH)
    vertices = mesh.vertices.astype(np.float32)
    faces = mesh.faces
    print(f"  頂点数: {len(vertices):,}")
    print(f"  面数: {len(faces):,}")

    # データベース作成
    print("\n[2/4] データベース作成...")
    conn = create_database(DB_PATH)
    cursor = conn.cursor()

    # メタデータ保存
    import json
    metadata = {
        'model': str(MODEL_PATH),
        'render_size': json.dumps(list(RENDER_SIZE)),
        'feature_size': json.dumps(list(FEATURE_SIZE)),
        'rotation_order': 'ZYX',
        'angle_step': json.dumps({
            'theta': THETA_RANGE.step,
            'phi': PHI_RANGE.step,
            'roll': ROLL_RANGE.step
        })
    }
    for key, value in metadata.items():
        cursor.execute('INSERT INTO metadata (key, value) VALUES (?, ?)', (key, value))
    conn.commit()

    # 角度パターンを生成
    angle_patterns = []
    for theta in THETA_RANGE:
        for phi in PHI_RANGE:
            for roll in ROLL_RANGE:
                angle_patterns.append((theta, phi, roll, vertices, faces))

    total = len(angle_patterns)
    print(f"\n[3/4] 特徴量計算・保存... (全{total:,}パターン)")

    insert_sql = '''
        INSERT INTO features
        (theta, phi, roll, aspect_ratio, fill_ratio, centroid_x, centroid_y,
         hu1, hu2, hu3, hu4, silhouette)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    '''

    count = 0
    valid_count = 0
    batch = []
    batch_size = 5000
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
        futures = {executor.submit(process_angle, args): args for args in angle_patterns}

        for future in as_completed(futures):
            count += 1
            result = future.result()

            if result is not None:
                valid_count += 1
                row = (
                    result["theta"],
                    result["phi"],
                    result["roll"],
                    result["aspect_ratio"],
                    result["fill_ratio"],
                    result["centroid_x"],
                    result["centroid_y"],
                    result["hu_moments"][0],
                    result["hu_moments"][1],
                    result["hu_moments"][2],
                    result["hu_moments"][3],
                    result["silhouette"]
                )
                batch.append(row)

                if len(batch) >= batch_size:
                    cursor.executemany(insert_sql, batch)
                    conn.commit()
                    batch = []

            if count % 10000 == 0:
                elapsed = time.time() - start_time
                eta = elapsed / count * (total - count)
                print(f"  進捗: {count:,}/{total:,} ({count/total*100:.1f}%) - "
                      f"経過: {elapsed:.0f}秒, 残り: {eta:.0f}秒")

    # 残りを保存
    if batch:
        cursor.executemany(insert_sql, batch)
        conn.commit()

    # total_patternsを保存
    cursor.execute('INSERT INTO metadata (key, value) VALUES (?, ?)',
                   ('total_patterns', str(valid_count)))
    conn.commit()

    # インデックス作成
    print("\n[4/4] インデックス作成・最適化...")
    cursor.execute('CREATE INDEX idx_angles ON features(theta, phi, roll)')
    cursor.execute('CREATE INDEX idx_aspect ON features(aspect_ratio)')
    cursor.execute('CREATE INDEX idx_fill ON features(fill_ratio)')
    cursor.execute('VACUUM')
    conn.close()

    # 結果表示
    db_size = DB_PATH.stat().st_size / 1024 / 1024
    total_time = time.time() - start_time

    print("\n" + "=" * 60)
    print("  完了！")
    print("=" * 60)
    print(f"  有効パターン: {valid_count:,}")
    print(f"  ファイルサイズ: {db_size:.1f} MB")
    print(f"  処理時間: {total_time:.1f}秒 ({total_time/60:.1f}分)")
    print("=" * 60)


if __name__ == "__main__":
    main()
