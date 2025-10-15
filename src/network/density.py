import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import ProjectionConfig
from .encoder import MultiViewProjectionEncoder, PointFeatureSampler, , PointFeatureSampler
from .fusion import PointWiseCrossViewFusion
from src.encoder import get_encoder


class PointWiseFusionDensityNet(nn.Module):
    """
    点级别融合的密度网络：
    多视图编码 → 各视图点特征采样 → 点级别跨视图融合 → 3D密度回归
    """

    def __init__(
        self,
        projection_sizes: dict[str, tuple[int, int]],
        encoder_type: str = "swin",
        feature_dims: list[int] | None = None,
        point_feature_dim: int = 32,
        fused_feature_dim: int = 256,
    ):
        super().__init__()

        self.config = ProjectionConfig(projection_sizes)
        self.encoder_type = encoder_type

        # 1. 多视图投影编码器
        self.projection_encoder = MultiViewProjectionEncoder(
            config=self.config, encoder_type=encoder_type, feature_dims=feature_dims
        )

        # 2. 点特征采样器
        self.point_feature_sampler = PointFeatureSampler(feature_dims)

        # 3. 点级别跨视图融合
        self.point_fusion = PointWiseCrossViewFusion(
            num_views=len(self.config.views),
            point_feature_dim=self.point_feature_sampler.total_output_dim,
            fused_feature_dim=fused_feature_dim,
        )

        # 4. Instant NGP 哈希编码器
        # self.hash_encoder = HashEncodingWrapper(**default_hash_config)
        self.hash_encoder = get_encoder("hashgrid")

        # 5. 密度回归MLP
        mlp_input_dim = self.hash_encoder.output_dim + fused_feature_dim
        self.density_mlp = nn.Sequential(
            nn.Linear(mlp_input_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

        print(f"PointWiseFusionDensityNet:")
        print(f"  Number of views: {len(self.config.views)}")
        print(
            f"  Point feature dim per view: {self.point_feature_sampler.total_output_dim}"
        )
        print(f"  Fused feature dim: {fused_feature_dim}")
        print(f"  Hash encoding dim: {self.hash_encoder.output_dim}")
        print(f"  MLP input dim: {mlp_input_dim}")

    def forward(
        self,
        projections: dict[str, torch.Tensor],
        points: torch.Tensor | None = None,
        return_point_features: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            projections: 多视图投影图像
            points: 3D点坐标，如果为None则自动生成
            return_point_features: 是否返回点特征（用于调试）
        """
        batch_size = next(iter(projections.values())).shape[0]

        # 1. 编码多视图投影
        encoder_output = self.projection_encoder(projections)

        # 2. 生成采样点（如果未提供）
        if points is None:
            points = self._generate_sampling_points(batch_size)

        # 3. 在各个视图上采样点特征
        view_point_features = {}
        for view_name in self.config.views:
            # 获取该视图的多尺度特征
            view_multi_scale = encoder_output["view_multi_scale_features"][view_name]

            # 采样该视图的点特征
            point_features = self.point_feature_sampler.sample_view_features(
                view_multi_scale,
                points,
                view_name,
                self.config.projection_sizes[view_name],
            )
            view_point_features[view_name] = point_features

        # 4. 点级别跨视图融合
        fused_point_features = self.point_fusion(view_point_features, points)

        # 5. 哈希编码3D坐标
        hash_features = self.hash_encoder(points)

        # 6. 拼接特征并回归密度
        combined_features = torch.cat([hash_features, fused_point_features], dim=-1)
        densities = self.density_mlp(combined_features)

        if return_point_features:
            return densities, {
                "view_point_features": view_point_features,
                "fused_point_features": fused_point_features,
                "hash_features": hash_features,
            }
        else:
            return densities

    def _generate_sampling_points(
        self, batch_size: int, num_points: int = 10000
    ) -> torch.Tensor:
        """生成采样点"""
        device = next(iter(self.parameters())).device
        points = torch.rand(batch_size, num_points, 3, device=device)
        return points

    def get_density_at_points(
        self, projections: dict[str, torch.Tensor], points: torch.Tensor
    ) -> torch.Tensor:
        """在指定点查询密度值"""
        return self.forward(projections, points)

    def reconstruct_voxel_grid(
        self,
        projections: dict[str, torch.Tensor],
        grid_size: tuple[int, int, int] = (64, 64, 64),
        chunk_size: int = 50000,
    ) -> torch.Tensor:
        """重建完整的体素网格"""
        batch_size = next(iter(projections.values())).shape[0]
        device = next(iter(projections.values())).device

        d, h, w = grid_size
        total_points = d * h * w

        # 初始化输出网格
        voxel_grid = torch.zeros(batch_size, d, h, w, device=device)

        # 分块处理
        for start_idx in range(0, total_points, chunk_size):
            end_idx = min(start_idx + chunk_size, total_points)

            # 生成当前块的网格点
            chunk_points = self._generate_grid_chunk(
                batch_size, grid_size, start_idx, end_idx
            )

            # 计算当前块的密度
            with torch.cuda.amp.autocast():
                chunk_densities = self.get_density_at_points(projections, chunk_points)

            # 将密度值放回网格
            chunk_densities = chunk_densities.view(batch_size, -1, 1, 1, 1)
            # 这里需要实现将扁平索引映射回3D网格的逻辑
            # 简化实现：假设按顺序排列
            for b in range(batch_size):
                voxel_grid[b].view(-1)[start_idx:end_idx] = chunk_densities[b].view(-1)

        return voxel_grid

    def _generate_grid_chunk(
        self,
        batch_size: int,
        grid_size: tuple[int, int, int],
        start_idx: int,
        end_idx: int,
    ) -> torch.Tensor:
        """生成网格块的点坐标"""
        d, h, w = grid_size

        # 创建网格索引
        indices = torch.arange(start_idx, end_idx)
        z = indices // (h * w)
        y = (indices % (h * w)) // w
        x = indices % w

        # 转换为归一化坐标
        z_coords = z.float() / (d - 1) if d > 1 else torch.zeros_like(z.float())
        y_coords = y.float() / (h - 1) if h > 1 else torch.zeros_like(y.float())
        x_coords = x.float() / (w - 1) if w > 1 else torch.zeros_like(x.float())

        chunk_points = torch.stack(
            [x_coords, y_coords, z_coords], dim=-1
        )  # [chunk_size, 3]
        chunk_points = chunk_points.unsqueeze(0).repeat(
            batch_size, 1, 1
        )  # [B, chunk_size, 3]

        return chunk_points
