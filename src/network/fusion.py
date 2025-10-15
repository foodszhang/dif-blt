import torch
import torch.nn as nn
import torch.nn.functional as F


class PointWiseCrossViewFusion(nn.Module):
    """
    点级别的跨视图融合：先在各个视图采样点特征，再用注意力融合
    """

    def __init__(
        self,
        num_views: int = 4,
        point_feature_dim: int = 32,  # 每个视图的点特征维度
        fused_feature_dim: int = 256,
        num_heads: int = 8,
    ):
        super().__init__()
        self.num_views = num_views
        self.point_feature_dim = point_feature_dim
        self.fused_feature_dim = fused_feature_dim

        # 1. 视图特定的点特征变换
        self.view_point_transforms = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(point_feature_dim, 64),
                    nn.ReLU(inplace=True),
                    nn.Linear(64, 128),
                    nn.ReLU(inplace=True),
                    nn.Linear(128, fused_feature_dim // num_heads),  # 为多头注意力准备
                )
                for _ in range(num_views)
            ]
        )

        # 2. 跨视图点特征注意力
        self.cross_view_attention = nn.MultiheadAttention(
            embed_dim=fused_feature_dim // num_heads,
            num_heads=num_heads,
            batch_first=True,
        )

        # 3. 特征融合与输出
        self.feature_fusion = nn.Sequential(
            nn.Linear(fused_feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, fused_feature_dim),
            nn.ReLU(inplace=True),
        )

        # 4. 位置编码（增强几何信息）
        self.position_encoding = nn.Sequential(
            nn.Linear(3, 64),  # 3D坐标
            nn.ReLU(inplace=True),
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, fused_feature_dim // num_heads),
        )

    def forward(
        self, view_point_features: dict[str, torch.Tensor], points: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            view_point_features: 字典 {view_name: [B, N, point_feature_dim]}
            points: [B, N, 3] 3D点坐标
        Returns:
            fused_point_features: [B, N, fused_feature_dim] 融合后的点特征
        """
        batch_size, num_points, _ = points.shape

        # 1. 对每个视图的点特征进行变换
        transformed_view_features = []
        for i, (view_name, point_features) in enumerate(view_point_features.items()):
            transformed = self.view_point_transforms[i](
                point_features
            )  # [B, N, fused_feature_dim//num_heads]
            transformed_view_features.append(transformed)

        # 2. 添加位置编码
        position_features = self.position_encoding(
            points
        )  # [B, N, fused_feature_dim//num_heads]

        # 3. 跨视图注意力融合
        fused_features = []
        for i in range(self.num_views):
            # 当前视图作为query
            query = transformed_view_features[i] + position_features  # [B, N, C]

            # 所有视图拼接作为key和value
            key = torch.cat(transformed_view_features, dim=1)  # [B, N*num_views, C]
            value = key

            # 注意力融合
            attended, _ = self.cross_view_attention(query, key, value)
            fused_features.append(attended)

        # 4. 合并所有视图的注意力结果
        fused = torch.stack(fused_features, dim=-1)  # [B, N, C, num_views]
        fused = fused.mean(dim=-1)  # [B, N, C] - 平均融合

        # 5. 最终特征变换
        fused_point_features = self.feature_fusion(fused)  # [B, N, fused_feature_dim]

        return fused_point_features
