"""
3Dモデルと2D画像のマッチング（IOU最大化）
GPU高速化版 - PyTorch + CUDA

使用方法:
    python match_3d_gpu.py

必要なライブラリ:
    pip install torch torchvision trimesh pillow matplotlib pycollada
"""

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

def load_target_image(image_path):
    """ターゲット画像を読み込み"""
    img = Image.open(image_path).convert('RGBA')
    img_array = np.array(img)
    gray = np.mean(img_array[:, :, :3], axis=2)
    mask = (gray > 15).astype(np.float32)
    return img_array, mask


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


def render_silhouette_gpu(vertices, faces, rotation_angles, image_size=(1920, 1080), scale_factor=0.65):
    """GPUでシルエットをレンダリング"""
    rx, ry, rz = rotation_angles
    R = create_rotation_matrix(rx, ry, rz)
    
    # 頂点を回転
    rotated = vertices @ R.T
    
    # 2D投影
    width, height = image_size
    min_dim = min(width, height)
    scale = min_dim * scale_factor
    
    proj_x = rotated[:, 0] * scale + width / 2
    proj_y = height / 2 - rotated[:, 1] * scale
    
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


# ===== GPU グリッドサーチ =====

def gpu_grid_search(mesh, target_mask_np, coarse_step=15, fine_step=3):
    """GPU高速グリッドサーチ"""
    # データをGPUに転送
    vertices = torch.tensor(mesh.vertices, device=DEVICE, dtype=torch.float32)
    faces = torch.tensor(mesh.faces, device=DEVICE, dtype=torch.long)
    target_mask = torch.tensor(target_mask_np, device=DEVICE, dtype=torch.float32)
    
    image_size = (target_mask_np.shape[1], target_mask_np.shape[0])
    
    print("\n  [Phase 1] GPU粗い探索中...")
    best_iou = 0
    best_angles = (0, 0, 0)
    
    rx_range = list(range(-180, 180, coarse_step))
    ry_range = list(range(-180, 180, coarse_step))
    
    total = len(rx_range) * len(ry_range)
    count = 0
    start_time = time.time()
    
    for rx in rx_range:
        for ry in ry_range:
            count += 1
            angles = (rx, ry, 0)
            
            silhouette = render_silhouette_gpu(vertices, faces, angles, image_size)
            iou = calculate_iou_gpu(silhouette, target_mask)
            
            if iou > best_iou:
                best_iou = iou
                best_angles = angles
                elapsed = time.time() - start_time
                print(f"    [{count}/{total}] 更新: ({rx:4d}, {ry:4d})° IOU={iou:.4f} ({elapsed:.1f}s)")
    
    print(f"\n  粗い探索結果: {best_angles}° → IOU={best_iou:.4f}")
    
    # Phase 2: 細かい探索
    print("\n  [Phase 2] GPU細かい探索中...")
    rx_center, ry_center, _ = best_angles
    
    for rx in range(rx_center - coarse_step, rx_center + coarse_step + 1, fine_step):
        for ry in range(ry_center - coarse_step, ry_center + coarse_step + 1, fine_step):
            angles = (rx, ry, 0)
            silhouette = render_silhouette_gpu(vertices, faces, angles, image_size)
            iou = calculate_iou_gpu(silhouette, target_mask)
            
            if iou > best_iou:
                best_iou = iou
                best_angles = angles
                print(f"    更新: ({rx:4d}, {ry:4d})° IOU={iou:.4f}")
    
    return best_angles, best_iou


# ===== 可視化 =====

def visualize_results(target_img, target_mask, mesh, best_angles, best_iou, output_path):
    """結果を可視化"""
    vertices = torch.tensor(mesh.vertices, device=DEVICE, dtype=torch.float32)
    faces = torch.tensor(mesh.faces, device=DEVICE, dtype=torch.long)
    
    image_size = (target_mask.shape[1], target_mask.shape[0])
    best_silhouette = render_silhouette_gpu(vertices, faces, best_angles, image_size)
    best_silhouette = best_silhouette.cpu().numpy()
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    
    axes[0, 0].imshow(target_img)
    axes[0, 0].set_title('Target Image', fontsize=14)
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(target_mask, cmap='gray')
    axes[0, 1].set_title('Target Mask', fontsize=14)
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(best_silhouette, cmap='gray')
    axes[0, 2].set_title(f'3D Model Silhouette\nAngles: ({best_angles[0]:.1f}°, '
                         f'{best_angles[1]:.1f}°, {best_angles[2]:.1f}°)', fontsize=14)
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
    # パス設定（修正済み）
    BASE_PATH = Path(r"C:\Users\tomoya\Contacts\デスクトップ\永野研3D")
    
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
    
    # 2. モデル読み込み
    print(f"\n[2/4] 3Dモデルを読み込み中...")
    mesh = load_3d_model(MODEL_PATH)
    print(f"   頂点数: {len(mesh.vertices):,}")
    print(f"   面数: {len(mesh.faces):,}")
    
    # 3. GPU最適化
    print(f"\n[3/4] GPU最適化中...")
    start = time.time()
    best_angles, best_iou = gpu_grid_search(mesh, target_mask)
    elapsed = time.time() - start
    
    print(f"\n結果:")
    print(f"   最適角度: X={best_angles[0]:.1f}°, Y={best_angles[1]:.1f}°, Z={best_angles[2]:.1f}°")
    print(f"   最大IOU: {best_iou:.4f} ({best_iou*100:.2f}%)")
    print(f"   処理時間: {elapsed:.1f}秒")
    
    # 4. 可視化
    print(f"\n[4/4] 結果を可視化中...")
    visualize_results(target_img, target_mask, mesh, best_angles, best_iou, OUTPUT_PATH)
    
    print("\n" + "=" * 70)
    print("  完了！")
    print("=" * 70)


if __name__ == "__main__":
    main()
