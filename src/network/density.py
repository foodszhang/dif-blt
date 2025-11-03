import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import ProjectionConfig
from .encoder import SpatialAttentionFusion, UNet, GateFusion


class PointDensityNet(nn.Module):
    """整体荧光光源重建网络：多视图U-Net特征提取 + 空间注意力融合 + 密度预测"""

    def __init__(self, num_views=4, in_channels=1, feature_dim=64, pos_enc_dim=18):
        super().__init__()
        self.pos_enc_dim = pos_enc_dim
        # 1. 多视图差异化特征提取器（每个视图独立U-Net）
        self.view_extractors = nn.ModuleList(
            [
                UNet(n_channels=in_channels, n_features=feature_dim, bilinear=True)
                for _ in range(num_views)
            ]
        )
        # self.view_extractors = UNet(
        #     n_channels=in_channels, n_features=feature_dim, bilinear=True
        # )

        # 2. 空间注意力融合模块
        self.attention_fusion = SpatialAttentionFusion(
            feature_dim=feature_dim, pos_enc_dim=pos_enc_dim
        )
        self.gate_fusions = nn.ModuleList(
            [GateFusion(feature_dim) for _ in range(num_views)]
        )

        # 3. 光源密度预测头（针对稀疏光源）
        self.density_head = nn.Sequential(
            # nn.Linear(feature_dim + pos_enc_dim, feature_dim),
            nn.Linear(feature_dim * 7, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=0.1),  # 抑制背景过拟合
            nn.Linear(feature_dim, feature_dim // 2),
            nn.BatchNorm1d(feature_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim // 2, 1),
            nn.Sigmoid(),  # 输出0-1的光源密度
        )

    def forward(self, view_projections, x3d):
        """
        view_projections: 列表，每个元素为[B, in_channels, H_i, W_i]（多视图投影图）
        x3d: [B, N, 3] 3D点坐标
        输出:
            density: [B, N, 1] 预测光源密度
            attention_weights: [B, N, num_views] 各视图注意力权重
        """
        # 1. 提取多视图特征（每个视图独立处理）
        view_list = ["-90", "-60", "-30", "0", "30", "60", "90"]
        view_features = {}
        views_no_projections = {}
        for view_name, projection in view_projections.items():
            ###TODO: HARD CODE， 认定了view_name的dx形式
            angle = view_name
            projection = projection.unsqueeze(1)
            feat = self.view_extractors[view_list.index(str(angle))](projection)
            # feat = self.view_extractors(projection)
            view_features[view_name] = feat
            views_no_projections[view_name] = self.gate_fusions[
                view_list.index(str(angle))
            ](feat)
            views_no_projections[view_name] = views_no_projections[view_name].squeeze(1)

        # 2. 空间注意力融合
        # fused_feat, attn_weights, pos_enc = self.attention_fusion(
        #     x3d, view_features
        # )  # [B, N, feature_dim], [B, N, num_views]
        fused_feat = self.attention_fusion(x3d, view_features)

        # 3. 预测光源密度
        B, N, C = fused_feat.shape
        # total_feat = torch.cat([fused_feat, pos_enc], dim=-1)
        total_feat = fused_feat
        # density = self.density_head(
        #     total_feat.reshape(B * N, C + self.pos_enc_dim)
        # ).view(B, N, 1)
        density = self.density_head(total_feat.reshape(B * N, C)).view(B, N, 1)
        # density = self.density_head(pos_enc).view(B, N, 1)

        # return density, attn_weights
        return density, views_no_projections
