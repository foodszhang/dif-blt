import torch
import torch.nn as nn
import torch.nn.functional as F
from timm import create_model
import numpy as np
from .base import ProjectionConfig


class BaseEncoder(nn.Module):
    """基础编码器接口"""

    def __init__(self, in_channels=1, feature_dims: list[int] | None = None):
        super().__init__()
        self.in_channels = in_channels
        self.feature_dims = (
            feature_dims if feature_dims is not None else [64, 128, 256, 512]
        )

    def forward(self, x):
        raise NotImplementedError


class SwinEncoder(BaseEncoder):
    """Swin Transformer编码器"""

    def __init__(
        self,
        in_channels=1,
        encoder_name="swin_tiny_patch4_window7_224",
        feature_dims: list[int] | None = None,
    ):
        super().__init__(in_channels, feature_dims)

        self.encoder = create_model(
            encoder_name,
            pretrained=False,
            in_chans=in_channels,
            features_only=True,
        )

        original_channels = self.encoder.feature_info.channels()

        self.feature_transforms = nn.ModuleList()
        for i, (orig_dim, target_dim) in enumerate(
            zip(original_channels, self.feature_dims)
        ):
            if orig_dim != target_dim:
                transform = nn.Sequential(
                    nn.Conv2d(orig_dim, target_dim, 1),
                    nn.BatchNorm2d(target_dim),
                    nn.ReLU(inplace=True),
                )
            else:
                transform = nn.Identity()
            self.feature_transforms.append(transform)

        self.encoder_channels = self.feature_dims

    def forward(self, x):
        features = self.encoder(x)
        transformed_features = []

        for i, feat in enumerate(features):
            transformed = self.feature_transforms[i](feat)
            transformed_features.append(transformed)

        return transformed_features


class UNetEncoder(BaseEncoder):
    """原始U-Net编码器"""

    def __init__(self, in_channels=1, feature_dims: List[int] = None):
        super().__init__(in_channels, feature_dims)

        if feature_dims is None:
            feature_dims = [64, 128, 256, 512, 1024]

        self.enc1 = self._block(in_channels, feature_dims[0])
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = self._block(feature_dims[0], feature_dims[1])
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = self._block(feature_dims[1], feature_dims[2])
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = self._block(feature_dims[2], feature_dims[3])
        self.pool4 = nn.MaxPool2d(2)
        self.enc5 = self._block(feature_dims[3], feature_dims[4])

        self.encoder_channels = feature_dims

    def _block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        features = []

        x1 = self.enc1(x)
        features.append(x1)
        x2 = self.enc2(self.pool1(x1))
        features.append(x2)
        x3 = self.enc3(self.pool2(x2))
        features.append(x3)
        x4 = self.enc4(self.pool3(x3))
        features.append(x4)
        x5 = self.enc5(self.pool4(x4))
        features.append(x5)

        return features


class MultiViewProjectionEncoder(nn.Module):
    """
    多视图投影编码器：更好地支持点采样回归
    """

    def __init__(
        self,
        config: ProjectionConfig,
        encoder_type: str = "swin",
        feature_dims: list[int] | None = None,
        unified_size: tuple[int, int] = (64, 64),
    ):
        super().__init__()
        self.config = config
        self.encoder_type = encoder_type
        self.unified_size = unified_size

        if feature_dims is None:
            if encoder_type == "swin":
                feature_dims = [64, 128, 256, 512]
            else:
                feature_dims = [64, 128, 256, 512, 1024]

        self.feature_dims = feature_dims

        # 为每个投影面创建独立的编码器
        self.view_encoders = nn.ModuleDict()

        for view_name in config.views:
            if encoder_type == "swin":
                encoder = SwinEncoder(
                    in_channels=config.projection_channels, feature_dims=feature_dims
                )
            else:
                encoder = UNetEncoder(
                    in_channels=config.projection_channels, feature_dims=feature_dims
                )
            self.view_encoders[view_name] = encoder

        # 多尺度特征采样器（用于点采样回归）
        self.multiscale_sampler = MultiScaleFeatureSampler(feature_dims)

        # 自适应池化层
        self.adaptive_pools = nn.ModuleList(
            [nn.AdaptiveAvgPool2d(unified_size) for _ in range(len(feature_dims))]
        )

        # 特征融合卷积（用于跨视图融合）
        self.fusion_convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(dim, 64, 3, padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                )
                for dim in feature_dims
            ]
        )

        self.multiscale_feature_dim = 64 * len(feature_dims)

    def forward(self, projections: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """返回多尺度融合特征和每个视图的原始多尺度特征"""
        multi_scale_features = []  # 用于跨视图融合
        view_multi_scale_features = {}  # 用于点采样回归

        # 处理每个投影面
        for view_name, proj_tensor in projections.items():
            encoder = self.view_encoders[view_name]

            # 获取原始多尺度特征（用于点采样）
            raw_multi_scale = encoder(proj_tensor)
            view_multi_scale_features[view_name] = raw_multi_scale

            # 多尺度特征融合（用于跨视图融合）
            view_features = []
            for i, (feat, pool, conv) in enumerate(
                zip(raw_multi_scale, self.adaptive_pools, self.fusion_convs)
            ):
                pooled_feat = pool(feat)
                fused_feat = conv(pooled_feat)
                view_features.append(fused_feat)

            view_multiscale = torch.cat(view_features, dim=1)
            multi_scale_features.append(view_multiscale)

        return {
            "multiscale_features": multi_scale_features,  # 用于CrossViewFusion
            "view_multi_scale_features": view_multi_scale_features,  # 用于PointSamplingDecoder
        }


class PointFeatureSampler(nn.Module):
    """
    点特征采样器：从各个视图的特征图中采样点特征
    """

    def __init__(self, feature_dims: List[int], output_dims: List[int] = None):
        super().__init__()
        self.feature_dims = feature_dims
        self.num_scales = len(feature_dims)

        # 设置每个尺度的输出维度
        if output_dims is None:
            self.output_dims = [max(8, dim // 8) for dim in feature_dims]
        else:
            self.output_dims = output_dims

        # 多尺度特征变换
        self.scale_transforms = nn.ModuleList()
        for i, (in_dim, out_dim) in enumerate(zip(feature_dims, self.output_dims)):
            if in_dim != out_dim:
                transform = nn.Sequential(
                    nn.Conv2d(in_dim, out_dim, 1),  # 1x1卷积降维
                    nn.BatchNorm2d(out_dim),
                    nn.ReLU(inplace=True),
                )
            else:
                transform = nn.Identity()
            self.scale_transforms.append(transform)

        # 计算总特征维度
        self.total_output_dim = sum(self.output_dims)

    def sample_view_features(
        self,
        multi_scale_features: List[torch.Tensor],
        points: torch.Tensor,
        view_name: str,
        original_size: Tuple[int, int],
    ) -> torch.Tensor:
        """
        从单个视图的多尺度特征图中采样点特征
        """
        batch_size, num_points, _ = points.shape
        device = points.device

        # 存储所有尺度的点特征
        scale_point_features = []

        for scale_idx, (features, transform) in enumerate(
            zip(multi_scale_features, self.scale_transforms)
        ):
            # 变换特征图
            transformed_features = transform(features)  # [B, C_out, H, W]
            _, feat_dim, feat_h, feat_w = transformed_features.shape

            # 将3D点投影到当前视图
            proj_coords = self.project_points_to_view(
                points, view_name, original_size, (feat_h, feat_w)
            )

            # 双线性插值采样特征
            point_feat = (
                F.grid_sample(
                    transformed_features,
                    proj_coords.unsqueeze(2).unsqueeze(2),
                    mode="bilinear",
                    align_corners=True,
                    padding_mode="border",
                )
                .squeeze(-1)
                .squeeze(-1)
            )  # [B, C, N]

            point_feat = point_feat.transpose(1, 2)  # [B, N, C]
            scale_point_features.append(point_feat)

        # 拼接所有尺度的特征
        view_point_features = torch.cat(
            scale_point_features, dim=-1
        )  # [B, N, total_output_dim]

        return view_point_features

    def project_points_to_view(
        self,
        points: torch.Tensor,
        view_name: str,
        original_size: Tuple[int, int],
        feature_size: Tuple[int, int],
    ):
        """将3D点投影到2D视图坐标"""
        batch_size, num_points, _ = points.shape
        orig_h, orig_w = original_size
        feat_h, feat_w = feature_size

        # 根据视图名称选择投影方式
        if view_name == "d1":
            proj_coords = points[:, :, :2]  # [B, N, 2]
        elif view_name == "d2":
            proj_coords = points[:, :, :2]  # [B, N, 2]
        elif view_name == "d3":
            proj_coords = points[:, :, 1:]  # [B, N, 2]
        elif view_name == "d4":
            proj_coords = points[:, :, 1:]  # [B, N, 2]
        else:
            raise ValueError(f"Unknown view name: {view_name}")

        # 坐标转换
        proj_coords[:, :, 0] = proj_coords[:, :, 0] * (feat_w - 1)
        proj_coords[:, :, 1] = proj_coords[:, :, 1] * (feat_h - 1)

        proj_coords[:, :, 0] = 2.0 * proj_coords[:, :, 0] / (feat_w - 1) - 1.0
        proj_coords[:, :, 1] = 2.0 * proj_coords[:, :, 1] / (feat_h - 1) - 1.0

        return proj_coords


class MultiScaleFeatureSampler(nn.Module):
    """
    多尺度特征采样器：从所有尺度特征图中采样点特征
    """

    def __init__(self, feature_dims: list[int], output_dims: list[int] | None = None):
        super().__init__()
        self.feature_dims = feature_dims
        self.num_scales = len(feature_dims)

        # 设置每个尺度的输出维度
        if output_dims is None:
            # 默认：为每个尺度设置不同的输出维度，浅层特征分配更多通道
            self.output_dims = [max(16, dim // 4) for dim in feature_dims]
            # 调整使得浅层特征有更多表达能力
            self.output_dims[0] = max(32, self.output_dims[0])  # 最浅层
            self.output_dims[1] = max(24, self.output_dims[1])  # 次浅层
        else:
            self.output_dims = output_dims

        # 为每个尺度创建特征变换网络
        self.scale_transforms = nn.ModuleList()
        for i, (in_dim, out_dim) in enumerate(zip(feature_dims, self.output_dims)):
            if in_dim != out_dim:
                transform = nn.Sequential(
                    nn.Conv2d(in_dim, out_dim, 3, padding=1),
                    nn.BatchNorm2d(out_dim),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(out_dim, out_dim, 1),  # 最终投影
                )
            else:
                transform = nn.Identity()
            self.scale_transforms.append(transform)

        # 计算总特征维度
        self.total_output_dim = sum(self.output_dims)

    def project_points_to_view(
        self,
        points: torch.Tensor,
        view_name: str,
        original_size: tuple[int, int],
        feature_size: tuple[int, int],
    ):
        """
        将3D点投影到2D视图坐标
        points: [B, N, 3] 在[0,1]范围内
        return: [B, N, 2] 在[-1,1]范围内的网格采样坐标
        """
        batch_size, num_points, _ = points.shape

        # 根据视图名称选择投影方式
        if view_name == "d1":
            proj_coords = points[:, :, :2]  # [B, N, 2]
        elif view_name == "d2":
            proj_coords = points[:, :, :2]  # [B, N, 2]
        elif view_name == "d3":
            proj_coords = points[:, :, 1:]  # [B, N, 2]
        elif view_name == "d4":
            proj_coords = points[:, :, 1:]  # [B, N, 2]
        else:
            raise ValueError(f"Unknown view name: {view_name}")

        # 从[0,1]坐标转换到特征图坐标
        orig_h, orig_w = original_size
        feat_h, feat_w = feature_size

        # 缩放坐标到特征图尺寸
        proj_coords[:, :, 0] = proj_coords[:, :, 0] * (feat_w - 1)
        proj_coords[:, :, 1] = proj_coords[:, :, 1] * (feat_h - 1)

        # 转换到[-1,1]范围 (grid_sample要求的范围)
        proj_coords[:, :, 0] = 2.0 * proj_coords[:, :, 0] / (feat_w - 1) - 1.0
        proj_coords[:, :, 1] = 2.0 * proj_coords[:, :, 1] / (feat_h - 1) - 1.0

        return proj_coords

    def sample_features(
        self,
        multi_scale_features: List[torch.Tensor],
        points: torch.Tensor,
        projection_sizes: Dict[str, Tuple[int, int]],
    ) -> Dict[str, torch.Tensor]:
        """
        从多尺度特征图中采样点特征

        Args:
            multi_scale_features: 每个尺度的特征图列表
            points: [B, N, 3] 3D点坐标
            projection_sizes: 各视图的原始尺寸

        Returns:
            scale_point_features: 每个尺度的点特征字典
        """
        batch_size, num_points, _ = points.shape
        scale_point_features = {}

        for scale_idx, (features, transform) in enumerate(
            zip(multi_scale_features, self.scale_transforms)
        ):
            # 变换特征图
            transformed_features = transform(features)  # [B, C_out, H, W]
            _, feat_dim, feat_h, feat_w = transformed_features.shape

            # 存储所有视图在当前尺度的特征
            scale_features = []

            for view_name, original_size in projection_sizes.items():
                # 将3D点投影到当前视图
                proj_coords = self.project_points_to_view(
                    points, view_name, original_size, (feat_h, feat_w)
                )

                # 双线性插值采样特征
                point_feat = (
                    F.grid_sample(
                        transformed_features,
                        proj_coords.unsqueeze(2).unsqueeze(2),
                        mode="bilinear",
                        align_corners=True,
                    )
                    .squeeze(-1)
                    .squeeze(-1)
                )  # [B, C, N]

                point_feat = point_feat.transpose(1, 2)  # [B, N, C]
                scale_features.append(point_feat)

            # 拼接所有视图在当前尺度的特征
            scale_combined = torch.cat(scale_features, dim=-1)  # [B, N, V * C_out]
            scale_point_features[f"scale_{scale_idx}"] = scale_combined

        return scale_point_features
