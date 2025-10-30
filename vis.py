import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import gridspec
import os


def visualize_voxel_comparison(true_voxel, optimized_voxel, threshold=0.5):
    """
    可视化真实体素与优化体素的三维对比

    参数:
    true_voxel: 真实体素网格
    optimized_voxel: 优化得到的体素网格
    threshold: 阈值，大于该值的体素被视为"存在"
    """
    # 创建图形
    fig = plt.figure(figsize=(15, 10))
    gs = gridspec.GridSpec(2, 3, height_ratios=[1, 1])

    # 获取真实体素和优化体素的坐标（大于阈值的部分）
    true_x, true_y, true_z = np.where(true_voxel >= threshold)
    opt_x, opt_y, opt_z = np.where(optimized_voxel >= threshold)

    # 1. 真实体素三维可视化
    ax1 = fig.add_subplot(gs[0, 0], projection="3d")
    ax1.scatter(true_x, true_y, true_z, c="blue", s=1, alpha=0.8)
    ax1.set_title("真实体素网格")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Z")
    ax1.set_xlim(0, 49)
    ax1.set_ylim(0, 49)
    ax1.set_zlim(0, 49)

    # 2. 优化体素三维可视化
    ax2 = fig.add_subplot(gs[0, 1], projection="3d")
    ax2.scatter(opt_x, opt_y, opt_z, c="red", s=1, alpha=0.8)
    ax2.set_title("优化体素网格")
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.set_zlabel("Z")
    ax2.set_xlim(0, 49)
    ax2.set_ylim(0, 49)
    ax2.set_zlim(0, 49)

    # 3. 叠加对比
    ax3 = fig.add_subplot(gs[0, 2], projection="3d")
    ax3.scatter(true_x, true_y, true_z, c="blue", s=1, alpha=0.5, label="真实")
    ax3.scatter(opt_x, opt_y, opt_z, c="red", s=1, alpha=0.5, label="优化")
    ax3.set_title("叠加对比")
    ax3.set_xlabel("X")
    ax3.set_ylabel("Y")
    ax3.set_zlabel("Z")
    ax3.set_xlim(0, 49)
    ax3.set_ylim(0, 49)
    ax3.set_zlim(0, 49)
    ax3.legend()

    # 4. 误差可视化（绝对值）
    error = np.abs(optimized_voxel - true_voxel)
    err_x, err_y, err_z = np.where(error > 0)
    err_values = error[err_x, err_y, err_z]

    ax4 = fig.add_subplot(gs[1, :], projection="3d")
    scatter = ax4.scatter(
        err_x, err_y, err_z, c=err_values, cmap="viridis", s=2, alpha=0.8
    )
    ax4.set_title("体素误差分布（红色表示误差大）")
    ax4.set_xlabel("X")
    ax4.set_ylabel("Y")
    ax4.set_zlabel("Z")
    ax4.set_xlim(0, 49)
    ax4.set_ylim(0, 49)
    ax4.set_zlim(0, 49)
    cbar = plt.colorbar(scatter, ax=ax4, pad=0.1)
    cbar.set_label("误差值（绝对值）")

    plt.tight_layout()
    return fig


def visualize_projections(true_projections, optimized_projections):
    """可视化真实投影与优化体素生成的投影对比"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    projection_names = ["x=49 (Y-Z)", "y=49 (X-Z)", "z=49 (X-Y)"]

    for i in range(3):
        # 真实投影
        ax1 = axes[0, i]
        im1 = ax1.imshow(
            true_projections[i], cmap="gray", vmin=0, vmax=np.max(true_projections)
        )
        ax1.set_title(f"真实投影 {projection_names[i]}")
        ax1.set_xlabel("n")
        ax1.set_ylabel("m")
        plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

        # 优化投影
        ax2 = axes[1, i]
        im2 = ax2.imshow(
            optimized_projections[i], cmap="gray", vmin=0, vmax=np.max(true_projections)
        )
        ax2.set_title(f"优化投影 {projection_names[i]}")
        ax2.set_xlabel("n")
        ax2.set_ylabel("m")
        plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    plt.tight_layout()
    return fig


def compute_optimized_projections(optimized_voxel, contribution_matrix):
    """计算优化体素的投影值"""
    projections = np.zeros((3, 50, 50), dtype=np.float32)
    for i in range(3):
        projections[i] = np.einsum(
            "xyz,xyzmn->mn", optimized_voxel, contribution_matrix[i]
        )
    return projections


def show_results(result_path, contribution_path=None):
    """从保存的结果文件展示所有对比"""
    # 加载结果
    results = np.load(result_path)
    optimized_voxel = results["optimized_voxel"]
    true_voxel = results["true_voxel"]
    loss_history = results["loss_history"]

    # 加载贡献矩阵（用于计算优化体素的投影）
    if contribution_path:
        contribution_data = np.load(contribution_path)
        contribution_matrix = contribution_data["contribution_matrix"]
        true_projections = contribution_data["projections"]
        optimized_projections = compute_optimized_projections(
            optimized_voxel, contribution_matrix
        )
    else:
        true_projections = None
        optimized_projections = None

    # 1. 显示损失曲线
    plt.figure(figsize=(10, 5))
    plt.plot(loss_history)
    plt.title("训练损失曲线")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.show()

    # 2. 显示体素对比
    fig_voxel = visualize_voxel_comparison(true_voxel, optimized_voxel, threshold=0.5)
    plt.show()

    # 3. 显示投影对比（如果有数据）
    if true_projections is not None and optimized_projections is not None:
        fig_proj = visualize_projections(true_projections, optimized_projections)
        plt.show()

    # 打印统计信息
    print("体素统计信息:")
    print(f"真实体素中值为1的数量: {np.sum(true_voxel >= 0.5)}")
    print(f"优化体素中值大于0.5的数量: {np.sum(optimized_voxel >= 0.5)}")
    print(f"平均绝对误差: {np.mean(np.abs(optimized_voxel - true_voxel)):.6f}")
    print(f"均方误差: {np.mean((optimized_voxel - true_voxel) ** 2):.6f}")


if __name__ == "__main__":
    # 示例：展示结果（请替换为实际的结果文件路径）
    # 第一个参数：优化结果文件路径
    # 第二个参数：原始数据集文件路径（用于获取贡献矩阵和真实投影）
    show_results("results/voxel_optimization_results.npz", "voxel_sphere_dataset.npz")
