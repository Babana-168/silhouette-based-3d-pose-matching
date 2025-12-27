"""
test4.py - 元画像に3Dモデルをテクスチャ付きで合成（安定版 / TPSなし）

狙い
- test2で得た最適角度（rank01_*.png）を読み取る
- 3Dモデルをテクスチャ付きでpyrenderでオフスクリーンレンダリング
- 元画像から白うさぎ領域を抽出（最大連結成分）
- 3Dレンダ結果を自動スケール & 自動位置合わせして合成
- TPS(Thin Plate Spline)は使わない（破壊要因のため）

注意
- OpenGLオフスクリーン環境が必要
  例: LinuxならEGL/OSMesa、Windowsなら環境によって動作が変わる
"""

import os
from pathlib import Path
import numpy as np
import cv2
import trimesh
import pyrender


# ==========================
# 設定
# ==========================
BASE_DIR = Path(__file__).parent

# 入力（存在する方を自動選択）
INPUT_IMAGE = BASE_DIR / "Image0.png"

# モデルパス候補（どちらでも動くようにする）
MODEL_CANDIDATES = [
    BASE_DIR / "models_rabit_obj" / "rabit.obj",
    BASE_DIR / "rabit.obj",
]

# test2.py の結果ディレクトリ
RESULT_DIR = BASE_DIR / "rotation_results"

# 出力
OUTPUT_IMAGE = BASE_DIR / "final_result_warped.png"  # 名前はそのままでもよい
DEBUG_DIR = BASE_DIR / "debug_test4"
DEBUG_DIR.mkdir(exist_ok=True)

# 合成設定
MODEL_ALPHA = 0.7  # 0.0～1.0

# 微調整（必要ならここだけ触る）
ADJUST_THETA = 0.0
ADJUST_PHI   = 0.0
ADJUST_ROLL  = 0.0
ADJUST_SCALE = 1.0
ADJUST_X = 0
ADJUST_Y = 0

# OpenGLバックエンド設定（環境で効く/効かないあり）
# 先に egl を試し、ダメなら osmesa を試す
if "PYOPENGL_PLATFORM" not in os.environ:
    os.environ["PYOPENGL_PLATFORM"] = "egl"


# ==========================
# 角度取得
# ==========================
def get_best_angles_from_results(default=(0.0, 0.0, 0.0, 0.0)):
    """
    rotation_resultsフォルダから rank01 の角度を取得
    ファイル名例: rank01_theta+098.0_phi-088.0_roll+055.0_iou0.9138.png
    """
    if not RESULT_DIR.exists():
        return default

    files = sorted(RESULT_DIR.glob("rank01_*.png"))
    if not files:
        return default

    stem = files[0].stem
    parts = stem.split("_")
    try:
        theta = float(parts[1].replace("theta", ""))
        phi   = float(parts[2].replace("phi", ""))
        roll  = float(parts[3].replace("roll", ""))
        iou   = float(parts[4].replace("iou", ""))
        return theta, phi, roll, iou
    except Exception:
        return default


# ==========================
# 画像読み込み（Unicodeパス対応）
# ==========================
def imread_unicode(path: Path):
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return img


# ==========================
# 回転変換
# ==========================
def spherical_to_rotation(theta_deg, phi_deg, roll_deg=0.0, initial_rx=-90.0):
    """
    test2の角度系 -> 3D回転角へ変換
    """
    rx = initial_rx + phi_deg
    ry = -theta_deg
    rz = roll_deg
    return rx, ry, rz


# ==========================
# モデルパス解決
# ==========================
def resolve_model_path():
    for p in MODEL_CANDIDATES:
        if p.exists():
            return p
    raise FileNotFoundError(f"rabit.obj が見つからない: candidates={MODEL_CANDIDATES}")


# ==========================
# テクスチャ付きレンダリング
# ==========================
def render_model_textured(mesh_path: Path, theta, phi, roll, size_wh, scale_factor=0.8):
    import pyglet
    from pyglet import gl

    # trimeshロード
    mesh = trimesh.load(str(mesh_path), force="mesh", process=True)

    # 正規化
    mesh.vertices -= mesh.centroid
    ext = mesh.bounding_box.extents.max()
    if ext > 0:
        mesh.vertices /= ext

    # 回転
    rx, ry, rz = spherical_to_rotation(theta, phi, roll)
    Rz = trimesh.transformations.rotation_matrix(np.radians(rz), [0, 0, 1])
    Ry = trimesh.transformations.rotation_matrix(np.radians(ry), [0, 1, 0])
    Rx = trimesh.transformations.rotation_matrix(np.radians(rx), [1, 0, 0])
    mesh.apply_transform(Rx @ Ry @ Rz)


    mesh.vertices *= float(scale_factor)

    # scene
    scene = pyrender.Scene(bg_color=[0, 0, 0, 0], ambient_light=[0.8, 0.8, 0.8])
    pr_mesh = pyrender.Mesh.from_trimesh(mesh, smooth=True)
    scene.add(pr_mesh)

    camera = pyrender.OrthographicCamera(xmag=1.2, ymag=1.2)
    cam_pose = np.eye(4)
    cam_pose[2, 3] = 2.0
    scene.add(camera, pose=cam_pose)

    light = pyrender.DirectionalLight(color=[1, 1, 1], intensity=3.0)
    scene.add(light, pose=cam_pose)

    w, h = int(size_wh[0]), int(size_wh[1])

    # pyglet window
    config = gl.Config(double_buffer=False, depth_size=24, alpha_size=8)
    win = pyglet.window.Window(
        width=w,
        height=h,
        visible=False,
        config=config
    )

    # OpenGL初期化
    gl.glEnable(gl.GL_DEPTH_TEST)
    gl.glViewport(0, 0, w, h)
    gl.glClearColor(0, 0, 0, 0)
    gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

    # 描画
    # 描画
    r = pyrender.Renderer(viewport_width=w, viewport_height=h)
    r.render(scene, flags=pyrender.RenderFlags.RGBA)
    r.delete()


    # ---------
    # 色取得
    # ---------
    buffer = (gl.GLubyte * (w * h * 4))()
    gl.glReadPixels(
        0, 0, w, h,
        gl.GL_RGBA,
        gl.GL_UNSIGNED_BYTE,
        buffer
    )

    color_rgba = np.frombuffer(buffer, dtype=np.uint8).reshape(h, w, 4)
    color_rgba = np.flipud(color_rgba)  # OpenGL座標反転

    # ---------
    # 深度取得
    # ---------
    depth_buf = (gl.GLfloat * (w * h))()
    gl.glReadPixels(
        0, 0, w, h,
        gl.GL_DEPTH_COMPONENT,
        gl.GL_FLOAT,
        depth_buf
    )

    depth = np.frombuffer(depth_buf, dtype=np.float32).reshape(h, w)
    depth = np.flipud(depth)

    win.close()

    color_bgr = cv2.cvtColor(color_rgba, cv2.COLOR_RGBA2BGR)
    mask = (depth < 1.0).astype(np.uint8) * 255

    return color_bgr, mask



# ==========================
# 元画像マスク（最大連結成分）
# ==========================
def extract_white_mask(image_bgr):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    # うさぎが明るい前提で閾値
    _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

    # ノイズ除去
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return mask

    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    largest_mask = (labels == largest_label).astype(np.uint8) * 255
    return largest_mask


def remove_background(image_bgr):
    mask = extract_white_mask(image_bgr)
    # mask で切り抜き（背景は黒）
    result = cv2.bitwise_and(image_bgr, image_bgr, mask=mask)
    return result, mask


# ==========================
# bbox
# ==========================
def get_bbox(mask_u8):
    ys, xs = np.where(mask_u8 > 0)
    if xs.size == 0 or ys.size == 0:
        return None
    x_min, x_max = int(xs.min()), int(xs.max())
    y_min, y_max = int(ys.min()), int(ys.max())
    return {
        "x_min": x_min,
        "x_max": x_max,
        "y_min": y_min,
        "y_max": y_max,
        "center_x": (x_min + x_max) // 2,
        "center_y": (y_min + y_max) // 2,
        "width": x_max - x_min + 1,
        "height": y_max - y_min + 1,
    }


# ==========================
# 合成（自動スケール + 自動位置合わせ）
# ==========================
def composite_images(bg_removed, bg_mask, model_color, model_mask, alpha, manual_dx=0, manual_dy=0):
    h, w = bg_removed.shape[:2]

    # モデルを元画像サイズへ（まずは同サイズにしてbbox取る）
    model_color_rs = cv2.resize(model_color, (w, h), interpolation=cv2.INTER_LINEAR)
    model_mask_rs  = cv2.resize(model_mask,  (w, h), interpolation=cv2.INTER_NEAREST)

    bg_bbox = get_bbox(bg_mask)
    md_bbox = get_bbox(model_mask_rs)

    if bg_bbox is None or md_bbox is None:
        offset_x, offset_y = manual_dx, manual_dy
        model_color_final = model_color_rs
        model_mask_final  = model_mask_rs
    else:
        # 高さベースのスケール比
        scale_ratio = bg_bbox["height"] / max(1, md_bbox["height"])

        # 追加スケール
        scale_ratio *= float(ADJUST_SCALE)

        # スケール適用（元モデル画像サイズに対して）
        oh, ow = model_color.shape[:2]
        new_w = max(1, int(ow * scale_ratio))
        new_h = max(1, int(oh * scale_ratio))

        model_color_scaled = cv2.resize(model_color, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        model_mask_scaled  = cv2.resize(model_mask,  (new_w, new_h), interpolation=cv2.INTER_NEAREST)

        # キャンバスに配置（中央スタート）
        model_color_final = np.zeros((h, w, 3), dtype=np.uint8)
        model_mask_final  = np.zeros((h, w), dtype=np.uint8)

        start_x = (w - new_w) // 2
        start_y = (h - new_h) // 2

        # 範囲計算（はみ出し対策）
        dst_x0 = max(0, start_x)
        dst_y0 = max(0, start_y)
        dst_x1 = min(w, start_x + new_w)
        dst_y1 = min(h, start_y + new_h)

        src_x0 = max(0, -start_x)
        src_y0 = max(0, -start_y)
        src_x1 = src_x0 + (dst_x1 - dst_x0)
        src_y1 = src_y0 + (dst_y1 - dst_y0)

        if dst_x1 > dst_x0 and dst_y1 > dst_y0:
            model_color_final[dst_y0:dst_y1, dst_x0:dst_x1] = model_color_scaled[src_y0:src_y1, src_x0:src_x1]
            model_mask_final[dst_y0:dst_y1, dst_x0:dst_x1]  = model_mask_scaled[src_y0:src_y1, src_x0:src_x1]

        # スケール後bboxで自動位置合わせ
        md2 = get_bbox(model_mask_final)
        if md2 is None:
            offset_x, offset_y = manual_dx, manual_dy
        else:
            # Xは中心合わせ、Yは下端合わせ
            offset_x = (bg_bbox["center_x"] - md2["center_x"]) + int(manual_dx)
            offset_y = (bg_bbox["y_max"] - md2["y_max"]) + int(manual_dy)

    # オフセット適用
    if offset_x != 0 or offset_y != 0:
        M = np.float32([[1, 0, offset_x], [0, 1, offset_y]])
        model_color_final = cv2.warpAffine(model_color_final, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=0)
        model_mask_final  = cv2.warpAffine(model_mask_final,  M, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)

    # 合成領域は交差部分だけ
    combined_mask = cv2.bitwise_and(bg_mask, model_mask_final)

    # アルファ合成
    m = (combined_mask.astype(np.float32) / 255.0) * float(alpha)
    m3 = np.dstack([m, m, m])

    out = bg_removed.astype(np.float32) * (1.0 - m3) + model_color_final.astype(np.float32) * m3
    out = np.clip(out, 0, 255).astype(np.uint8)
    return out, model_color_final, model_mask_final, combined_mask


# ==========================
# main
# ==========================
def main():
    print("=" * 60)
    print("test4 安定版（TPSなし）")
    print("=" * 60)

    # 角度取得
    theta, phi, roll, iou = get_best_angles_from_results(default=(0.0, 0.0, 0.0, 0.0))
    theta += ADJUST_THETA
    phi   += ADJUST_PHI
    roll  += ADJUST_ROLL

    print(f"[angles] theta={theta:.1f}, phi={phi:.1f}, roll={roll:.1f}, iou={iou:.4f}")

    # 入力画像
    img = imread_unicode(INPUT_IMAGE)
    if img is None:
        raise FileNotFoundError(f"画像が見つからない: {INPUT_IMAGE}")
    h, w = img.shape[:2]
    print(f"[image] {w}x{h}")

    # 背景削除
    bg_removed, bg_mask = remove_background(img)

    # 3Dモデルレンダ
    model_path = resolve_model_path()
    print(f"[model] {model_path}")
    scale_factor = 0.8 * float(ADJUST_SCALE)
    model_color, model_mask = render_model_textured(model_path, theta, phi, roll, (w, h), scale_factor=scale_factor)

    # 合成
    result, model_color_final, model_mask_final, combined_mask = composite_images(
        bg_removed, bg_mask,
        model_color, model_mask,
        MODEL_ALPHA,
        manual_dx=ADJUST_X,
        manual_dy=ADJUST_Y,
    )

    # 保存
    cv2.imwrite(str(OUTPUT_IMAGE), result)
    print(f"[save] {OUTPUT_IMAGE}")

    # デバッグ出力
    cv2.imwrite(str(DEBUG_DIR / "bg_mask.png"), bg_mask)
    cv2.imwrite(str(DEBUG_DIR / "bg_removed.png"), bg_removed)
    cv2.imwrite(str(DEBUG_DIR / "model_color_raw.png"), model_color)
    cv2.imwrite(str(DEBUG_DIR / "model_mask_raw.png"), model_mask)
    cv2.imwrite(str(DEBUG_DIR / "model_color_final.png"), model_color_final)
    cv2.imwrite(str(DEBUG_DIR / "model_mask_final.png"), model_mask_final)
    cv2.imwrite(str(DEBUG_DIR / "combined_mask.png"), combined_mask)
    print(f"[debug] {DEBUG_DIR}")

    print("done")


if __name__ == "__main__":
    main()
