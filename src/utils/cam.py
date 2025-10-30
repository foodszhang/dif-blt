import numpy as np
import torch


def rotation_matrix_y(angle_deg):
    """绕Y轴旋转矩阵"""
    angle_rad = np.deg2rad(angle_deg)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)

    R = torch.asarray(
        [[cos_a, 0, sin_a], [0, 1.0, 0], [-sin_a, 0, cos_a]], dtype=torch.float32
    )
    return R


def world_to_camera_coords_batch_matrix(points, rotation_deg, camera_distance):
    """
    使用矩阵运算批量将世界坐标系中的点转换到相机坐标系

    参数:
    points: 批量点 [B, N, 3] 或 [N, 3]
    rotation_deg: 旋转角度
    camera_distance: 相机距离

    返回:
    camera_coords: 相机坐标系中的点 [B, N, 3] 或 [N, 3]
    """
    R = rotation_matrix_y(rotation_deg).to(points.device)

    if points.ndim == 3:
        B, N, _ = points.shape
        # 重塑为 [B*N, 3] 进行矩阵乘法
        points_flat = points.reshape(B * N, 3)
        camera_coords_flat = points_flat @ R.T  # 矩阵乘法
        # 重塑回 [B, N, 3]
        camera_coords = camera_coords_flat.reshape(B, N, 3)
    else:  # points.ndim == 2
        camera_coords = points @ R.T  # 直接矩阵乘法

    # 修正深度计算
    camera_coords[..., 2] = camera_distance - camera_coords[..., 2]

    return camera_coords


def project_points_to_camera(points, rotation_deg, camera_distance, detector_size):
    """
    使用矩阵运算批量投影点

    参数:
    points: 批量点 [B, N, 3] 或 [N, 3]
    rotation_deg: 旋转角度
    camera_distance: 相机距离
    detector_size: 探测器尺寸 (width, height)

    返回:
    projections: 投影坐标 [B, N, 2] 或 [N, 2]
    depths: 深度信息 [B, N, 2] 或 [N, 2] (深度值, 可见性标志)
    """
    # 使用矩阵运算转换坐标
    camera_coords = world_to_camera_coords_batch_matrix(
        points, rotation_deg, camera_distance
    )

    width, height = detector_size

    # 提取UV坐标和深度
    if points.ndim == 3:
        B, N, _ = points.shape
        u = camera_coords[:, :, 0]  # [B, N]
        v = camera_coords[:, :, 1]  # [B, N]
        depth_vals = camera_coords[:, :, 2]  # [B, N]

        # 计算可见性 (使用向量化操作)
        visible = (
            (torch.abs(u) <= width / 2)
            & (torch.abs(v) <= height / 2)
            & (depth_vals > 0)
        )

        # 构建投影坐标 [B, N, 2]
        projections = torch.zeros((B, N, 2), dtype=torch.float32, device=points.device)
        projections[:, :, 0] = u
        projections[:, :, 1] = v

        # 构建深度信息 [B, N, 2]
        depths = torch.zeros((B, N, 2), dtype=torch.float32, device=points.device)
        depths[:, :, 0] = depth_vals
        depths[:, :, 1] = visible.type(torch.float32)

    else:  # points.ndim == 2
        N, _ = points.shape
        u = camera_coords[:, 0]  # [N]
        v = camera_coords[:, 1]  # [N]
        depth_vals = camera_coords[:, 2]  # [N]

        # 计算可见性
        visible = (
            (torch.abs(u) <= width / 2)
            & (torch.abs(v) <= height / 2)
            & (depth_vals > 0)
        )

        # 构建投影坐标 [N, 2]
        projections = torch.zeros((N, 2), dtype=torch.float32, device=points.device)
        projections[:, 0] = u
        projections[:, 1] = v

        # 构建深度信息 [N, 2]
        depths = torch.zeros((N, 2), dtype=torch.float32, device=points.device)
        depths[:, 0] = depth_vals
        depths[:, 1] = visible.type(torch.float32)

    return projections, depths
