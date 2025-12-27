"""
test3.py - 元画像に3Dモデルをテクスチャ付きで合成
test2.pyで取得した最適角度を使用して、元画像に3Dモデルを重ねる
MTLファイルとテクスチャを使用してリアルにレンダリング
"""

import numpy as np
import cv2
import trimesh
from pathlib import Path
from PIL import Image
import pyrender
import os

# ==========================
# 設定
# ==========================
BASE_DIR = Path(__file__).parent

# 入力
INPUT_IMAGE = BASE_DIR / "Image0.png"
MODEL_PATH = BASE_DIR / "models_rabit_obj" / "rabit.obj"

# test2.pyの結果から最適角度を取得（手動設定またはファイルから読み込み）
# rotation_resultsフォルダのrank01ファイル名から角度を読み取る
RESULT_DIR = BASE_DIR / "rotation_results"

# 出力
OUTPUT_IMAGE = BASE_DIR / "final_result.png"

# レンダリング設定
RENDER_SIZE = (1920, 1080)  # 高解像度でレンダリング

# 微調整パラメータ（必要に応じて手動調整）
ADJUST_THETA = 0.0  # theta微調整（度）
ADJUST_PHI = 0.0    # phi微調整（度）
ADJUST_ROLL = 0.0   # roll微調整（度）
ADJUST_SCALE = 1.0  # スケール微調整（倍率）
ADJUST_X = 0        # X方向移動（ピクセル）
ADJUST_Y = 0        # Y方向移動（ピクセル）

# 合成設定
MODEL_ALPHA = 0.7  # 透明度（0.0～1.0）

# ==========================
# test2結果から角度を読み取る
# ==========================
def get_best_angles_from_results():
    """rotation_resultsフォルダからrank01の角度を取得"""
    result_files = sorted(RESULT_DIR.glob("rank01_*.png"))

    if not result_files:
        raise FileNotFoundError("rotation_resultsにrank01ファイルが見つかりません")

    # ファイル名から角度を抽出: rank01_theta+098.0_phi-088.0_roll+055.0_iou0.9138.png
    filename = result_files[0].stem
    parts = filename.split('_')

    theta = float(parts[1].replace('theta', ''))
    phi = float(parts[2].replace('phi', ''))
    roll = float(parts[3].replace('roll', ''))
    iou = float(parts[4].replace('iou', ''))

    return theta, phi, roll, iou

# ==========================
# 画像読み込み（日本語パス対応）
# ==========================
def imread_unicode(path: Path):
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)

# ==========================
# 回転変換
# ==========================
def spherical_to_rotation(theta_deg, phi_deg, roll_deg=0, initial_rx=-90):
    """データベースの角度を実際の回転角度に変換"""
    rx = initial_rx + phi_deg
    ry = -theta_deg
    rz = roll_deg
    return rx, ry, rz

# ==========================
# テクスチャ付きレンダリング（pyrender使用）
# ==========================
def render_model_textured(mesh_path, theta, phi, roll, size, scale_factor=0.8):
    """3Dモデルをテクスチャ付きでレンダリング"""
    # メッシュを読み込み（テクスチャ付き）
    mesh = trimesh.load(mesh_path, force='mesh', process=False)

    # 正規化
    mesh.vertices -= mesh.centroid
    mesh.vertices /= mesh.bounding_box.extents.max()

    # 回転を適用
    rx, ry, rz = spherical_to_rotation(theta, phi, roll)
    Rz = trimesh.transformations.rotation_matrix(np.radians(rz), [0, 0, 1])
    Ry = trimesh.transformations.rotation_matrix(np.radians(ry), [0, 1, 0])
    Rx = trimesh.transformations.rotation_matrix(np.radians(rx), [1, 0, 0])
    mesh.apply_transform(Rz)
    mesh.apply_transform(Ry)
    mesh.apply_transform(Rx)

    # スケール調整
    mesh.vertices *= scale_factor

    # pyrenderシーンを作成
    scene = pyrender.Scene(ambient_light=[1.0, 1.0, 1.0], bg_color=[0, 0, 0, 0])

    # メッシュをpyrender用に変換
    if hasattr(mesh.visual, 'material'):
        # テクスチャ付きマテリアル
        pr_mesh = pyrender.Mesh.from_trimesh(mesh, smooth=False)
    else:
        # フォールバック: 単色
        pr_mesh = pyrender.Mesh.from_trimesh(mesh, smooth=False)

    scene.add(pr_mesh)

    # カメラ設定（正射投影）
    camera = pyrender.OrthographicCamera(xmag=1.0, ymag=1.0)
    camera_pose = np.eye(4)
    camera_pose[2, 3] = 2.0  # Z距離
    scene.add(camera, pose=camera_pose)

    # ライト設定
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0)
    scene.add(light, pose=camera_pose)

    # レンダリング
    renderer = pyrender.OffscreenRenderer(size[0], size[1])
    color, depth = renderer.render(scene)
    renderer.delete()

    # マスク作成（深度から）
    mask = (depth > 0).astype(np.uint8) * 255

    # BGRからRGBに変換（OpenCV用）
    color_bgr = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)

    return color_bgr, mask

# ==========================
# 元画像のマスク抽出（最大連結成分のみ）
# ==========================
def extract_white_mask(image):
    """元画像から白いうさぎの部分を抽出（ノイズを除去して最大の領域のみ）"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

    # ノイズ除去
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # 連結成分分析で最大の領域のみを抽出
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

    if num_labels <= 1:
        # 何も検出されなかった場合
        return mask

    # 背景（ラベル0）を除いて、最大面積の連結成分を見つける
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])

    # 最大の連結成分のみのマスクを作成
    largest_mask = (labels == largest_label).astype(np.uint8) * 255

    print(f"  ノイズ除去: {num_labels - 1}個の領域から最大のもの（面積={stats[largest_label, cv2.CC_STAT_AREA]}px）を抽出")

    return largest_mask

# ==========================
# 背景を削除してうさぎのみ残す
# ==========================
def remove_background(image):
    """背景を削除して、うさぎのみを残す（背景は黒）"""
    mask = extract_white_mask(image)

    # マスクを3チャンネルに変換
    mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    # マスク領域のみを残し、背景は黒にする
    result = cv2.bitwise_and(image, mask_3ch)

    return result, mask

# ==========================
# バウンディングボックスを取得
# ==========================
def get_bbox(mask):
    """マスクからバウンディングボックスを取得"""
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return {
        'x_min': xs.min(),
        'x_max': xs.max(),
        'y_min': ys.min(),
        'y_max': ys.max(),
        'center_x': (xs.min() + xs.max()) // 2,
        'center_y': (ys.min() + ys.max()) // 2,
        'width': xs.max() - xs.min() + 1,
        'height': ys.max() - ys.min() + 1
    }

# ==========================
# 画像合成（背景削除済み画像に3Dモデルの色を被せる）
# ==========================
def composite_images(background_removed, bg_mask, model_color, model_mask, alpha, manual_offset_x=0, manual_offset_y=0):
    """背景削除済み画像に3Dモデルの色を透明度付きで被せる（スケール・位置自動調整）"""
    # サイズを合わせる
    h, w = background_removed.shape[:2]
    model_color_resized = cv2.resize(model_color, (w, h))
    model_mask_resized = cv2.resize(model_mask, (w, h))

    # バウンディングボックスを取得
    bg_bbox = get_bbox(bg_mask)
    model_bbox = get_bbox(model_mask_resized)

    if bg_bbox is None or model_bbox is None:
        print("  警告: バウンディングボックスを取得できませんでした")
        offset_x, offset_y = manual_offset_x, manual_offset_y
        model_color_final = model_color_resized
        model_mask_final = model_mask_resized
    else:
        # スケール比を計算（高さベース）
        bg_height = bg_bbox['height']
        model_height = model_bbox['height']
        scale_ratio = bg_height / model_height

        print(f"  スケール調整: {scale_ratio:.2f}x (元画像高さ={bg_height}px, 3Dモデル高さ={model_height}px)")

        # 3Dモデルをスケール調整（元のサイズからスケール）
        original_h, original_w = model_color.shape[:2]
        new_h = int(original_h * scale_ratio)
        new_w = int(original_w * scale_ratio)

        print(f"    元のサイズ: {original_w}x{original_h} -> スケール後: {new_w}x{new_h}")

        # スケール調整したサイズでリサイズ
        model_color_scaled = cv2.resize(model_color, (new_w, new_h))
        model_mask_scaled = cv2.resize(model_mask, (new_w, new_h))

        # 元のサイズのキャンバスに配置
        model_color_final = np.zeros((h, w, 3), dtype=np.uint8)
        model_mask_final = np.zeros((h, w), dtype=np.uint8)

        # 中央配置のための初期オフセット
        start_y = (h - new_h) // 2
        start_x = (w - new_w) // 2

        # ソースとデスティネーションの範囲を計算（負のオフセット対応）
        src_y_start = max(0, -start_y)
        src_x_start = max(0, -start_x)
        dst_y_start = max(0, start_y)
        dst_x_start = max(0, start_x)

        src_y_end = min(new_h, new_h + (h - (start_y + new_h)))
        src_x_end = min(new_w, new_w + (w - (start_x + new_w)))
        dst_y_end = min(h, start_y + new_h)
        dst_x_end = min(w, start_x + new_w)

        # 実際にコピーする範囲
        copy_h = min(dst_y_end - dst_y_start, src_y_end - src_y_start)
        copy_w = min(dst_x_end - dst_x_start, src_x_end - src_x_start)

        if copy_h > 0 and copy_w > 0:
            model_color_final[dst_y_start:dst_y_start+copy_h, dst_x_start:dst_x_start+copy_w] = \
                model_color_scaled[src_y_start:src_y_start+copy_h, src_x_start:src_x_start+copy_w]
            model_mask_final[dst_y_start:dst_y_start+copy_h, dst_x_start:dst_x_start+copy_w] = \
                model_mask_scaled[src_y_start:src_y_start+copy_h, src_x_start:src_x_start+copy_w]
            print(f"    配置: dst[{dst_y_start}:{dst_y_start+copy_h}, {dst_x_start}:{dst_x_start+copy_w}] <- src[{src_y_start}:{src_y_start+copy_h}, {src_x_start}:{src_x_start+copy_w}]")

        # スケール後のバウンディングボックスを再取得
        model_bbox_scaled = get_bbox(model_mask_final)

        if model_bbox_scaled is not None:
            # バウンディングボックスの下端（y_max）を合わせる
            offset_y_bottom = bg_bbox['y_max'] - model_bbox_scaled['y_max']

            # 中心のX座標を合わせる
            offset_x_center = bg_bbox['center_x'] - model_bbox_scaled['center_x']

            offset_x = offset_x_center + manual_offset_x
            offset_y = offset_y_bottom + manual_offset_y

            print(f"  位置調整: X={offset_x:.0f}px, Y={offset_y:.0f}px")
        else:
            offset_x, offset_y = manual_offset_x, manual_offset_y

    # オフセット適用
    if offset_x != 0 or offset_y != 0:
        M = np.float32([[1, 0, offset_x], [0, 1, offset_y]])
        model_color_final = cv2.warpAffine(model_color_final, M, (w, h))
        model_mask_final = cv2.warpAffine(model_mask_final, M, (w, h))

    # 元画像のマスクと3Dモデルのマスクの交差部分のみで合成
    combined_mask = cv2.bitwise_and(bg_mask, model_mask_final)

    # アルファブレンディング
    mask_normalized = combined_mask.astype(float) / 255.0 * alpha
    mask_3ch = np.stack([mask_normalized] * 3, axis=-1)

    # 合成: 背景削除済み画像をベースに、マスク領域に3Dモデルの色を重ねる
    result = background_removed.copy().astype(float)
    result = result * (1 - mask_3ch) + model_color_final.astype(float) * mask_3ch

    return result.astype(np.uint8)

# ==========================
# メイン処理
# ==========================
def main():
    print("=" * 60)
    print("3Dモデル合成（test2結果使用）")
    print("=" * 60)

    # test2の結果から最適角度を取得
    print("\n[1/5] test2結果から最適角度を取得中...")
    theta, phi, roll, iou = get_best_angles_from_results()
    print(f"  最適角度: Theta={theta:.1f}°, Phi={phi:.1f}°, Roll={roll:.1f}°")
    print(f"  IoU: {iou:.4f}")

    # 微調整を適用
    theta += ADJUST_THETA
    phi += ADJUST_PHI
    roll += ADJUST_ROLL
    scale_factor = 0.8 * ADJUST_SCALE

    if ADJUST_THETA != 0 or ADJUST_PHI != 0 or ADJUST_ROLL != 0:
        print(f"\n  微調整適用後: Theta={theta:.1f}°, Phi={phi:.1f}°, Roll={roll:.1f}°")
    if ADJUST_SCALE != 1.0:
        print(f"  スケール調整: {ADJUST_SCALE}x")

    # 元画像を読み込み
    print("\n[2/5] 元画像読み込み中...")
    original_image = imread_unicode(INPUT_IMAGE)
    if original_image is None:
        raise FileNotFoundError(f"画像が見つかりません: {INPUT_IMAGE}")
    print(f"  画像サイズ: {original_image.shape[1]}x{original_image.shape[0]}")

    # 背景を削除
    print("\n[3/5] 背景削除中...")
    background_removed, bg_mask = remove_background(original_image)
    print("  背景を黒に置き換えました")

    # 3Dモデルをテクスチャ付きでレンダリング
    print("\n[4/5] 3Dモデルレンダリング中（テクスチャ付き）...")
    size = (original_image.shape[1], original_image.shape[0])
    model_color, model_mask = render_model_textured(
        MODEL_PATH, theta, phi, roll, size, scale_factor
    )
    print(f"  レンダリングサイズ: {size[0]}x{size[1]}")
    print(f"  テクスチャ: rabit01.jpg")
    print(f"  透明度: {MODEL_ALPHA}")

    # 画像合成
    print("\n[5/5] 画像合成中...")
    result = composite_images(
        background_removed, bg_mask, model_color, model_mask,
        MODEL_ALPHA, ADJUST_X, ADJUST_Y
    )

    # 保存
    cv2.imwrite(str(OUTPUT_IMAGE), result)
    print(f"\n結果を保存: {OUTPUT_IMAGE}")

    # 統計情報
    model_pixels = np.sum(model_mask > 0)
    total_pixels = model_mask.size
    coverage = model_pixels / total_pixels * 100
    print(f"\n統計:")
    print(f"  3Dモデルカバー率: {coverage:.2f}%")

    print("\n" + "=" * 60)
    print("完了！")
    print("=" * 60)
    print(f"\n微調整が必要な場合は、test3.py内の以下のパラメータを編集してください：")
    print(f"  ADJUST_THETA  = {ADJUST_THETA}  # 角度微調整")
    print(f"  ADJUST_PHI    = {ADJUST_PHI}")
    print(f"  ADJUST_ROLL   = {ADJUST_ROLL}")
    print(f"  ADJUST_SCALE  = {ADJUST_SCALE}  # スケール調整")
    print(f"  ADJUST_X      = {ADJUST_X}      # 位置調整")
    print(f"  ADJUST_Y      = {ADJUST_Y}")
    print(f"  MODEL_ALPHA   = {MODEL_ALPHA}   # 透明度")

if __name__ == "__main__":
    main()
