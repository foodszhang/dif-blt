import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum
import math


class SpatialViewAttention(nn.Module):
    """
    基于空间注意力的多视图特征融合模块
    输入:
        - 3D点坐标: [B, N, 3]
        - 多视图特征图: List of [B, C, H_i, W_i]
        - 相机参数: 用于将3D点投影到各视图
    输出:
        - 3D点特征: [B, N, C]
        - 视图注意力权重: [B, N, V] (V为视图数量)
    """

    def __init__(self, feature_dim=64, num_heads=8, hidden_dim=256, dropout=0.1):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim

        # 空间位置编码
        self.pos_encoder = nn.Sequential(
            nn.Linear(3, 128), nn.ReLU(), nn.Linear(128, feature_dim)
        )

        # 注意力机制
        self.query_proj = nn.Linear(feature_dim, hidden_dim)
        self.key_proj = nn.Conv2d(feature_dim, hidden_dim, 1)
        self.value_proj = nn.Conv2d(feature_dim, feature_dim, 1)

        # 视图关系建模
        self.view_relation = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )

        # 输出投影
        self.output_proj = nn.Linear(feature_dim, feature_dim)
        self.dropout = nn.Dropout(dropout)

        # 层归一化
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(feature_dim)

    def project_3d_to_2d(self, points_3d, camera_params):
        """
        将3D点投影到2D图像坐标
        points_3d: [B, N, 3]
        camera_params: 包含内参和外参的字典
        返回: 各视图的2D坐标 [B, N, V, 2]
        """
        batch_size, num_points = points_3d.shape[:2]
        num_views = len(camera_params)

        # 简化的投影实现 - 实际应用中需要真实的相机参数
        projected_coords = []
        for view_idx in range(num_views):
            # 这里应该使用真实的相机投影矩阵
            # 简化版本: 假设已经预处理得到标准化坐标
            R = camera_params[view_idx]["rotation"]  # [B, 3, 3]
            T = camera_params[view_idx]["translation"]  # [B, 3]
            K = camera_params[view_idx]["intrinsic"]  # [B, 3, 3]

            # 世界坐标系到相机坐标系
            points_cam = einsum("bni,bij->bnj", points_3d, R) + T.unsqueeze(1)

            # 相机坐标系到图像坐标系
            points_img = einsum("bni,bij->bnj", points_cam, K)

            # 归一化到[-1, 1]范围 (PyTorch网格采样要求)
            # 这里需要根据实际图像尺寸进行调整
            coords = points_img[..., :2] / points_img[..., 2:3]
            projected_coords.append(coords.unsqueeze(2))

        return torch.cat(projected_coords, dim=2)  # [B, N, V, 2]

    def extract_view_features(self, feature_maps, coords):
        """
        使用双线性采样从特征图中提取对应位置的特征
        feature_maps: List of [B, C, H_i, W_i]
        coords: [B, N, V, 2] 归一化坐标
        返回: [B, N, V, C]
        """
        batch_size, num_points, num_views = coords.shape[:3]
        view_features = []

        for view_idx in range(num_views):
            feature_map = feature_maps[view_idx]  # [B, C, H, W]
            view_coords = coords[:, :, view_idx, :]  # [B, N, 2]

            # 调整坐标形状以适应grid_sample
            view_coords = view_coords.unsqueeze(1)  # [B, 1, N, 2]

            # 双线性采样
            sampled_features = F.grid_sample(
                feature_map,
                view_coords,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            )  # [B, C, 1, N]

            sampled_features = sampled_features.squeeze(2).transpose(1, 2)  # [B, N, C]
            view_features.append(sampled_features.unsqueeze(2))

        return torch.cat(view_features, dim=2)  # [B, N, V, C]

    def forward(self, points_3d, view_feature_maps, camera_params):
        batch_size, num_points = points_3d.shape[:2]
        num_views = len(view_feature_maps)

        # 1. 投影3D点到各视图
        coords_2d = self.project_3d_to_2d(points_3d, camera_params)

        # 2. 提取各视图对应位置的特征
        view_features = self.extract_view_features(
            view_feature_maps, coords_2d
        )  # [B, N, V, C]

        # 3. 添加3D位置编码
        pos_encoding = self.pos_encoder(points_3d)  # [B, N, C]
        pos_encoding = pos_encoding.unsqueeze(2).expand(-1, -1, num_views, -1)

        # 4. 准备注意力输入
        query = self.query_proj(pos_encoding)  # [B, N, V, hidden_dim]
        query = query.view(batch_size * num_points, num_views, self.hidden_dim)

        # 处理各视图特征作为key和value
        view_features_flat = view_features.view(
            batch_size * num_points, num_views, self.feature_dim
        )

        # 5. 视图间注意力 - 建模视图关系
        attended_features, attention_weights = self.view_relation(
            query, view_features_flat, view_features_flat
        )

        # 6. 残差连接与归一化
        attended_features = self.norm1(attended_features + query)
        attended_features = self.output_proj(attended_features)
        attended_features = self.norm2(attended_features)

        # 7.  reshape回原始维度
        final_features = attended_features.view(
            batch_size, num_points, num_views, self.feature_dim
        )
        attention_weights = attention_weights.view(
            batch_size, num_points, num_views, num_views
        )

        # 8. 聚合视图特征 (使用注意力加权的平均值)
        view_attention = torch.softmax(
            attention_weights.mean(dim=-1), dim=-1
        )  # [B, N, V]
        aggregated_features = einsum("bnvc,bnv->bnc", final_features, view_attention)

        return aggregated_features, view_attention


class SimpleAverageFusion(nn.Module):
    """
    简单的平均池化融合模块 - 用于消融实验
    """

    def __init__(self, feature_dim=64):
        super().__init__()
        self.feature_dim = feature_dim

    def project_3d_to_2d(self, points_3d, camera_params):
        # 与SpatialViewAttention相同的投影逻辑
        batch_size, num_points = points_3d.shape[:2]
        num_views = len(camera_params)

        projected_coords = []
        for view_idx in range(num_views):
            R = camera_params[view_idx]["rotation"]
            T = camera_params[view_idx]["translation"]
            K = camera_params[view_idx]["intrinsic"]

            points_cam = einsum("bni,bij->bnj", points_3d, R) + T.unsqueeze(1)
            points_img = einsum("bni,bij->bnj", points_cam, K)
            coords = points_img[..., :2] / points_img[..., 2:3]
            projected_coords.append(coords.unsqueeze(2))

        return torch.cat(projected_coords, dim=2)

    def extract_view_features(self, feature_maps, coords):
        batch_size, num_points, num_views = coords.shape[:3]
        view_features = []

        for view_idx in range(num_views):
            feature_map = feature_maps[view_idx]
            view_coords = coords[:, :, view_idx, :]
            view_coords = view_coords.unsqueeze(1)

            sampled_features = F.grid_sample(
                feature_map,
                view_coords,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            )

            sampled_features = sampled_features.squeeze(2).transpose(1, 2)
            view_features.append(sampled_features.unsqueeze(2))

        return torch.cat(view_features, dim=2)

    def forward(self, points_3d, view_feature_maps, camera_params):
        coords_2d = self.project_3d_to_2d(points_3d, camera_params)
        view_features = self.extract_view_features(view_feature_maps, coords_2d)

        # 简单平均池化
        aggregated_features = view_features.mean(dim=2)  # [B, N, C]
        uniform_attention = (
            torch.ones_like(view_features[..., 0]) / view_features.shape[2]
        )

        return aggregated_features, uniform_attention


# 测试代码
if __name__ == "__main__":
    # 模拟输入数据
    batch_size, num_points, num_views = 2, 100, 4
    feature_dim = 64

    # 模拟3D点
    points_3d = torch.randn(batch_size, num_points, 3)

    # 模拟不同尺寸的视图特征图
    view_feature_maps = [
        torch.randn(batch_size, feature_dim, 256, 256),  # 视图1
        torch.randn(batch_size, feature_dim, 128, 128),  # 视图2
        torch.randn(batch_size, feature_dim, 320, 240),  # 视图3
        torch.randn(batch_size, feature_dim, 192, 192),  # 视图4
    ]

    # 模拟相机参数
    camera_params = []
    for i in range(num_views):
        camera_params.append(
            {
                "rotation": torch.eye(3).unsqueeze(0).repeat(batch_size, 1, 1),
                "translation": torch.zeros(batch_size, 3),
                "intrinsic": torch.eye(3).unsqueeze(0).repeat(batch_size, 1, 1),
            }
        )

    # 测试注意力模块
    attention_model = SpatialViewAttention(feature_dim=feature_dim)
    features_att, attention_weights = attention_model(
        points_3d, view_feature_maps, camera_params
    )

    print(f"注意力模块输出特征形状: {features_att.shape}")
    print(f"注意力权重形状: {attention_weights.shape}")

    # 测试平均池化模块
    avg_model = SimpleAverageFusion(feature_dim=feature_dim)
    features_avg, avg_attention = avg_model(points_3d, view_feature_maps, camera_params)

    print(f"平均池化输出特征形状: {features_avg.shape}")
    print(f"平均注意力形状: {avg_attention.shape}")
