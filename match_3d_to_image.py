"""
3Dモデルと2D画像のマッチング（IOU最大化）
GPU高速化版 - PyTorch + CUDA

使用方法:
    python match_3d_gpu.py

必要なライブラリ:
    pip install torch torchvision trimesh pillow matplotlib pycollada
"""

from typing import Optional

import numpy as np
from PIL import Image, ImageDraw
import trimesh
import torch
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
import time
warnings.filterwarnings('ignore')

# GPU確認
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# ===== 画像・モデル読み込み =====

def otsu_threshold(gray: np.ndarray) -> float:
    """Otsu法で自動閾値を求める"""
    hist, bin_edges = np.histogram(gray.flatten(), bins=256, range=(0, 255))
    total = gray.size
    current_max, threshold = 0, 0
    sum_total = np.dot(np.arange(256), hist)
    sum_bg, weight_bg = 0.0, 0.0

    for i in range(256):
        weight_bg += hist[i]
        if weight_bg == 0:
            continue
        weight_fg = total - weight_bg
        if weight_fg == 0:
            break

        sum_bg += i * hist[i]
        mean_bg = sum_bg / weight_bg
        mean_fg = (sum_total - sum_bg) / weight_fg

        between_class_variance = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
        if between_class_variance > current_max:
            current_max = between_class_variance
            threshold = i

    return threshold


def load_target_image(image_path, threshold: Optional[float] = None, use_otsu: bool = True):
    """ターゲット画像を読み込み"""
    img = Image.open(image_path).convert('RGBA')
    img_array = np.array(img)
    gray = np.mean(img_array[:, :, :3], axis=2)

    if threshold is None and use_otsu:
        threshold = otsu_threshold(gray)
    elif threshold is None:
        threshold = 15

    mask = (gray > threshold).astype(np.float32)
    return img_array, mask


def downsample_mask(mask: np.ndarray, ratio: float) -> np.ndarray:
    """探索用にマスクをダウンサンプルする"""
    if ratio >= 1.0:
        return mask

    h, w = mask.shape
    new_size = (max(1, int(w * ratio)), max(1, int(h * ratio)))
    resized = (
        Image.fromarray((mask * 255).astype(np.uint8))
        .resize(new_size, resample=Image.NEAREST)
        .convert("L")
    )
    return (np.array(resized) > 0).astype(np.float32)


def load_3d_model(model_path):
    """3Dモデルを読み込む"""
    print(f"   モデルファイル: {model_path}")
    mesh = trimesh.load(model_path, force='mesh')
    
    if isinstance(mesh, trimesh.Scene):
        meshes = [g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if meshes:
            mesh = trimesh.util.concatenate(meshes)
    
    mesh.vertices -= mesh.centroid
    scale = 1.0 / mesh.bounding_box.extents.max()
    mesh.vertices *= scale
    
    return mesh


# ===== GPU レンダリング =====

def create_rotation_matrix(rx, ry, rz):
    """回転行列を作成（GPU上）"""
    rx_rad = torch.tensor(rx * np.pi / 180, device=DEVICE, dtype=torch.float32)
    ry_rad = torch.tensor(ry * np.pi / 180, device=DEVICE, dtype=torch.float32)
    rz_rad = torch.tensor(rz * np.pi / 180, device=DEVICE, dtype=torch.float32)
    
    # X軸回転
    Rx = torch.tensor([
        [1, 0, 0],
        [0, torch.cos(rx_rad).item(), -torch.sin(rx_rad).item()],
        [0, torch.sin(rx_rad).item(), torch.cos(rx_rad).item()]
    ], device=DEVICE, dtype=torch.float32)
    
    # Y軸回転
    Ry = torch.tensor([
        [torch.cos(ry_rad).item(), 0, torch.sin(ry_rad).item()],
        [0, 1, 0],
        [-torch.sin(ry_rad).item(), 0, torch.cos(ry_rad).item()]
    ], device=DEVICE, dtype=torch.float32)
    
    # Z軸回転
    Rz = torch.tensor([
        [torch.cos(rz_rad).item(), -torch.sin(rz_rad).item(), 0],
        [torch.sin(rz_rad).item(), torch.cos(rz_rad).item(), 0],
        [0, 0, 1]
    ], device=DEVICE, dtype=torch.float32)
    
    return Rz @ Ry @ Rx


def render_silhouette_gpu(
    vertices,
    faces,
    rotation_angles,
    image_size=(1920, 1080),
    scale_factor=0.65,
    translation=(0.0, 0.0),
):
    """GPUでシルエットをレンダリング"""
    rx, ry, rz = rotation_angles
    R = create_rotation_matrix(rx, ry, rz)

    # 頂点を回転
    rotated = vertices @ R.T

    # 2D投影
    width, height = image_size
    min_dim = min(width, height)
    scale = min_dim * scale_factor

    tx, ty = translation
    proj_x = rotated[:, 0] * scale + width / 2 + tx
    proj_y = height / 2 - rotated[:, 1] * scale + ty

    # CPUに戻してPILで描画
    proj_x_np = proj_x.cpu().numpy()
    proj_y_np = proj_y.cpu().numpy()
    faces_np = faces.cpu().numpy()

    img = Image.new('L', (width, height), 0)
    draw = ImageDraw.Draw(img)

    # 深度ソート
    z_order = rotated[faces].mean(dim=1)[:, 2].cpu().numpy()
    sorted_indices = np.argsort(z_order)

    for idx in sorted_indices:
        face = faces_np[idx]
        polygon = [(proj_x_np[face[i]], proj_y_np[face[i]]) for i in range(3)]
        draw.polygon(polygon, fill=255)

    silhouette = np.array(img) > 127
    return torch.tensor(silhouette, device=DEVICE, dtype=torch.float32)


def calculate_iou_gpu(mask1, mask2):
    """GPU上でIoU計算"""
    intersection = (mask1 * mask2).sum()
    union = ((mask1 + mask2) > 0).float().sum()
    return (intersection / union).item() if union > 0 else 0.0


# ===== サーチスペース計算と実行時間見積り =====

def compute_search_space(
    coarse_step=10,
    refine_window=10,
    fine_step=1,
    coarse_z_step=10,
    coarse_scale_factors=(0.55, 0.65, 0.75),
    translation_fractions=(-0.08, -0.04, 0.0, 0.04, 0.08),
    fine_scale_window=0.1,
):
    """粗探索・細探索の試行回数を返す"""

    # 粗探索（10度刻みの全軸回転）
    rx_range = list(range(-180, 181, coarse_step))
    ry_range = list(range(-180, 181, coarse_step))
    rz_range = list(range(-180, 181, coarse_z_step))
    coarse_total = len(rx_range) * len(ry_range) * len(rz_range)

    # 微探索（粗探索の最適角度を中心に±10°を1°刻みで走査）
    fine_rx_len = len(range(-refine_window, refine_window + 1, fine_step))
    fine_ry_len = fine_rx_len
    fine_rz_len = len(range(-refine_window, refine_window + 1, fine_step))
    refine_total = fine_rx_len * fine_ry_len * fine_rz_len

    # 角度が固まったあとに行うスケール/平行移動探索
    phase3_total = len(coarse_scale_factors) * len(translation_fractions) ** 2
    fine_scale_count = 9  # np.linspace(..., num=9)
    fine_translation_count = 9  # 3x3 グリッド
    phase4_total = fine_scale_count * fine_translation_count

    return {
        "coarse": coarse_total,
        "refine": refine_total,
        "scale": phase3_total,
        "fine_scale": phase4_total,
        "total": coarse_total + refine_total + phase3_total + phase4_total,
    }


def benchmark_iteration_time(
    vertices,
    faces,
    target_mask,
    image_size,
    samples=5,
    scale_factor=0.65,
    translation=(0.0, 0.0),
):
    """レンダリング+IoU計算の平均時間を簡易計測"""
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start = time.time()
    for i in range(samples):
        angle = (0, 0, (i * 30) % 180)
        silhouette = render_silhouette_gpu(vertices, faces, angle, image_size, scale_factor, translation)
        _ = calculate_iou_gpu(silhouette, target_mask)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    elapsed = time.time() - start
    return elapsed / samples if samples > 0 else 0.0


# ===== GPU グリッドサーチ =====

def gpu_grid_search(
    mesh,
    target_mask_np,
    translation_scale=1.0,
    coarse_step=10,
    refine_window=10,
    fine_step=1,
    coarse_z_step=10,
    coarse_scale_factors=(0.55, 0.65, 0.75),
    fine_scale_window=0.1,
    translation_fractions=(-0.08, -0.04, 0.0, 0.04, 0.08),
):
    """GPU高速グリッドサーチ（10°粗探索→±10°微探索→スケール/平行移動）"""
    vertices = torch.tensor(mesh.vertices, device=DEVICE, dtype=torch.float32)
    faces = torch.tensor(mesh.faces, device=DEVICE, dtype=torch.long)
    target_mask = torch.tensor(target_mask_np, device=DEVICE, dtype=torch.float32)

    image_size = (target_mask_np.shape[1], target_mask_np.shape[0])
    width, height = image_size
    min_dim = min(width, height)

    space = compute_search_space(
        coarse_step,
        refine_window,
        fine_step,
        coarse_z_step,
        coarse_scale_factors,
        translation_fractions,
        fine_scale_window,
    )
    print(
        "\n  サーチスペース:"
        f" 粗探索={space['coarse']:,}回"
        f", 微探索={space['refine']:,}回"
        f", スケール/平行移動={space['scale']:,}回"
        f", 微調整={space['fine_scale']:,}回"
        f", 合計={space['total']:,}回"
    )

    est_time = benchmark_iteration_time(vertices, faces, target_mask, image_size)
    if est_time > 0:
        print(f"  1試行あたり平均{est_time:.4f}秒（{int(space['total']*est_time)}秒程度を想定）")

    print("\n  [Phase 1] GPU粗い探索中 (回転のみ)...")
    best_iou = 0
    best_params = {
        "angles": (0, 0, 0),
        "scale": 0.65,
        "translation": (0.0, 0.0),  # 探索画像サイズ基準のpx
    }

    rx_range = list(range(-180, 181, coarse_step))
    ry_range = list(range(-180, 181, coarse_step))
    rz_range = list(range(-180, 181, coarse_z_step))

    total = len(rx_range) * len(ry_range) * len(rz_range)
    count = 0
    start_time = time.time()

    for rx in rx_range:
        for ry in ry_range:
            for rz in rz_range:
                count += 1
                angles = (rx, ry, rz)

                silhouette = render_silhouette_gpu(
                    vertices, faces, angles, image_size, best_params["scale"], best_params["translation"]
                )
                iou = calculate_iou_gpu(silhouette, target_mask)

                if iou > best_iou:
                    best_iou = iou
                    best_params["angles"] = angles
                    elapsed = time.time() - start_time
                    print(
                        f"    [{count}/{total}] 更新: ({rx:4d}, {ry:4d}, {rz:4d})° "
                        f"IOU={iou:.4f} ({elapsed:.1f}s)"
                    )

    print(
        f"\n  粗い探索結果: {best_params['angles']}° → IOU={best_iou:.4f}\n"
        "  [Phase 2] GPU微探索中 (最良角度±10°を1°刻み)..."
    )

    rx_center, ry_center, rz_center = best_params["angles"]
    fine_rx = range(rx_center - refine_window, rx_center + refine_window + 1, fine_step)
    fine_ry = range(ry_center - refine_window, ry_center + refine_window + 1, fine_step)
    fine_rz = range(rz_center - refine_window, rz_center + refine_window + 1, fine_step)

    for rx in fine_rx:
        for ry in fine_ry:
            for rz in fine_rz:
                angles = (rx, ry, rz)
                silhouette = render_silhouette_gpu(
                    vertices, faces, angles, image_size, best_params["scale"], best_params["translation"]
                )
                iou = calculate_iou_gpu(silhouette, target_mask)

                if iou > best_iou:
                    best_iou = iou
                    best_params["angles"] = angles
                    print(
                        f"    更新: ({rx:4d}, {ry:4d}, {rz:4d})° "
                        f"scale={best_params['scale']:.3f}, tx={best_params['translation'][0]*translation_scale:.1f}px, "
                        f"ty={best_params['translation'][1]*translation_scale:.1f}px IOU={iou:.4f}"
                    )

    print(
        f"\n  角度固定: {best_params['angles']}° → IOU={best_iou:.4f}\n"
        "  [Phase 3] スケール/平行移動を探索中..."
    )

    for scale in coarse_scale_factors:
        for tx_frac in translation_fractions:
            for ty_frac in translation_fractions:
                tx = tx_frac * min_dim
                ty = ty_frac * min_dim
                silhouette = render_silhouette_gpu(
                    vertices, faces, best_params["angles"], image_size, scale, (tx, ty)
                )
                iou = calculate_iou_gpu(silhouette, target_mask)

                if iou > best_iou:
                    best_iou = iou
                    best_params = {
                        "angles": best_params["angles"],
                        "scale": scale,
                        "translation": (tx, ty),
                    }
                    print(
                        f"    更新: ({best_params['angles'][0]:4d}, {best_params['angles'][1]:4d}, {best_params['angles'][2]:4d})° "
                        f"scale={scale:.3f}, tx={tx*translation_scale:.1f}px, ty={ty*translation_scale:.1f}px IOU={iou:.4f}"
                    )

    # さらなる微調整: スケールと平行移動を連続値で探索
    scale_center = best_params["scale"]
    fine_scales = np.linspace(
        max(0.1, scale_center - fine_scale_window), scale_center + fine_scale_window, num=9
    )

    tx_center, ty_center = best_params["translation"]
    fine_translations = [
        (tx_center + dx * min_dim, ty_center + dy * min_dim)
        for dx in (-0.02, 0.0, 0.02)
        for dy in (-0.02, 0.0, 0.02)
    ]

    print("\n  [Phase 4] スケール/平行移動の微調整...")

    for scale in fine_scales:
        for tx, ty in fine_translations:
            silhouette = render_silhouette_gpu(
                vertices, faces, best_params["angles"], image_size, scale, (tx, ty)
            )
            iou = calculate_iou_gpu(silhouette, target_mask)
            if iou > best_iou:
                best_iou = iou
                best_params = {
                    "angles": best_params["angles"],
                    "scale": float(scale),
                    "translation": (float(tx), float(ty)),
                }
                print(
                    f"    微調整: ({best_params['angles'][0]:4d}, {best_params['angles'][1]:4d}, {best_params['angles'][2]:4d})° "
                    f"scale={scale:.3f}, tx={tx*translation_scale:.1f}px, ty={ty*translation_scale:.1f}px IOU={iou:.4f}"
                )

    translated = (
        best_params["translation"][0] * translation_scale,
        best_params["translation"][1] * translation_scale,
    )
    return best_params["angles"], best_params["scale"], translated, best_iou


# ===== 可視化 =====

def visualize_results(target_img, target_mask, mesh, best_params, best_iou, output_path):
    """結果を可視化"""
    vertices = torch.tensor(mesh.vertices, device=DEVICE, dtype=torch.float32)
    faces = torch.tensor(mesh.faces, device=DEVICE, dtype=torch.long)

    image_size = (target_mask.shape[1], target_mask.shape[0])
    best_silhouette = render_silhouette_gpu(
        vertices,
        faces,
        best_params["angles"],
        image_size,
        best_params["scale"],
        best_params["translation"],
    )
    best_silhouette = best_silhouette.cpu().numpy()
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    
    axes[0, 0].imshow(target_img)
    axes[0, 0].set_title('Target Image', fontsize=14)
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(target_mask, cmap='gray')
    axes[0, 1].set_title('Target Mask', fontsize=14)
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(best_silhouette, cmap='gray')
    axes[0, 2].set_title(
        '3D Model Silhouette\n'
        f'Angles: ({best_params["angles"][0]:.1f}°, {best_params["angles"][1]:.1f}°, {best_params["angles"][2]:.1f}°)\n'
        f'Scale: {best_params["scale"]:.3f}, '
        f'Translation: ({best_params["translation"][0]:.1f}px, {best_params["translation"][1]:.1f}px)',
        fontsize=14,
    )
    axes[0, 2].axis('off')
    
    # オーバーレイ
    overlay = np.zeros((*target_mask.shape, 3), dtype=np.uint8)
    overlay[:, :, 0] = (target_mask * 255).astype(np.uint8)
    overlay[:, :, 1] = (best_silhouette * 255).astype(np.uint8)
    axes[1, 0].imshow(overlay)
    axes[1, 0].set_title(f'Overlay (Red=Target, Green=3D)\nIOU: {best_iou:.4f} ({best_iou*100:.2f}%)', fontsize=14)
    axes[1, 0].axis('off')
    
    # 差分
    diff = np.abs(target_mask - best_silhouette)
    axes[1, 1].imshow(diff, cmap='hot')
    axes[1, 1].set_title('Difference', fontsize=14)
    axes[1, 1].axis('off')
    
    # 交差
    intersection = target_mask * best_silhouette
    axes[1, 2].imshow(intersection, cmap='gray')
    axes[1, 2].set_title(f'Intersection\n{int(intersection.sum()):,} pixels', fontsize=14)
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n結果を保存: {output_path}")
    plt.show()


# ===== メイン =====

def main():
    # パス設定（リポジトリ直下のファイルを使用）
    BASE_PATH = Path(__file__).resolve().parent

    TARGET_IMAGE = BASE_PATH / "Image0.png"
    MODEL_PATH = BASE_PATH / "models_rabit_dae" / "rabit.dae"
    OUTPUT_PATH = BASE_PATH / "matching_result_gpu.png"
    
    print("=" * 70)
    print("  3Dモデル-画像マッチング（GPU高速化版）")
    print("=" * 70)
    
    # 1. 画像読み込み
    print(f"\n[1/4] ターゲット画像を読み込み中...")
    target_img, target_mask = load_target_image(TARGET_IMAGE)
    print(f"   サイズ: {target_img.shape[1]} x {target_img.shape[0]}")

    # 探索用ダウンサンプリング（高速化）
    SEARCH_DOWNSAMPLE = 0.5  # 50%に縮小して探索（1/4ピクセル数）
    search_mask = target_mask
    translation_scale = 1.0
    if SEARCH_DOWNSAMPLE < 1.0:
        search_mask = downsample_mask(target_mask, SEARCH_DOWNSAMPLE)
        translation_scale = 1.0 / SEARCH_DOWNSAMPLE
        print(
            f"   探索用に{int(SEARCH_DOWNSAMPLE*100)}%へ縮小: "
            f"{search_mask.shape[1]} x {search_mask.shape[0]} (translation x{translation_scale:.1f}で補正)"
        )
    
    # 2. モデル読み込み
    print(f"\n[2/4] 3Dモデルを読み込み中...")
    mesh = load_3d_model(MODEL_PATH)
    print(f"   頂点数: {len(mesh.vertices):,}")
    print(f"   面数: {len(mesh.faces):,}")
    
    # 3. GPU最適化
    print(f"\n[3/4] GPU最適化中...")
    start = time.time()
    best_angles, best_scale, best_translation, best_iou = gpu_grid_search(
        mesh,
        search_mask,
        translation_scale=translation_scale,
    )
    elapsed = time.time() - start

    print(f"\n結果:")
    print(f"   最適角度: X={best_angles[0]:.1f}°, Y={best_angles[1]:.1f}°, Z={best_angles[2]:.1f}°")
    print(f"   最適スケール: {best_scale:.3f}")
    print(f"   最適平行移動: tx={best_translation[0]:.1f}px, ty={best_translation[1]:.1f}px")
    print(f"   最大IOU: {best_iou:.4f} ({best_iou*100:.2f}%)")
    print(f"   処理時間: {elapsed:.1f}秒")

    # 4. 可視化
    print(f"\n[4/4] 結果を可視化中...")
    best_params = {
        "angles": best_angles,
        "scale": best_scale,
        "translation": best_translation,
    }
    visualize_results(target_img, target_mask, mesh, best_params, best_iou, OUTPUT_PATH)
    
    print("\n" + "=" * 70)
    print("  完了！")
    print("=" * 70)


if __name__ == "__main__":
    main()