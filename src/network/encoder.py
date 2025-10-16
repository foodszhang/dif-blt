import torch
import torch.nn as nn
import torch.nn.functional as F
from timm import create_model
import numpy as np
from .base import ProjectionConfig
from src.utils.utils import index_2d


class UNetLikeDecoder(nn.Module):
    """
    U-Net风格的解码器：将编码器特征图上采样回原始投影图尺寸
    """

    def __init__(
        self,
        encoder_channels: list[int],
        output_channels: int = 64,
        output_size: tuple[int, int] | None = None,
    ):
        super().__init__()
        self.encoder_channels = encoder_channels
        self.output_channels = output_channels
        self.output_size = output_size

        # 反转编码器通道顺序（从深层到浅层）
        decoder_channels = encoder_channels[::-1]

        # 上采样块
        self.up_blocks = nn.ModuleList()

        # 第一个上采样块（没有跳跃连接）
        self.up_blocks.append(UpBlock(decoder_channels[0], decoder_channels[1] // 2))

        # 后续上采样块（有跳跃连接）
        for i in range(1, len(decoder_channels) - 1):
            in_channels = decoder_channels[i] + decoder_channels[i + 1]  # 跳跃连接
            out_channels = (
                decoder_channels[i + 1] // 2
                if i < len(decoder_channels) - 2
                else output_channels
            )
            self.up_blocks.append(UpBlock(in_channels, out_channels))

        # 最终上采样到目标尺寸（如果需要）
        if output_size is not None:
            self.final_upsample = nn.Sequential(
                nn.Upsample(size=output_size, mode="bilinear", align_corners=True),
                nn.Conv2d(output_channels, output_channels, 3, padding=1),
                nn.BatchNorm2d(output_channels),
                nn.ReLU(inplace=True),
            )
        else:
            self.final_upsample = nn.Identity()

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            features: 编码器特征列表 [feat1, feat2, feat3, ...] 从浅到深
        Returns:
            decoded: 解码后的特征图 [B, C, H, W] 接近原始投影图尺寸
        """
        # 反转特征顺序（从深到浅）
        features = features[::-1]

        x = features[0]  # 最深层特征

        for i, up_block in enumerate(self.up_blocks):
            if i == 0:
                # 第一个块没有跳跃连接
                x = up_block(x)
                # pass
            else:
                # 后续块有跳跃连接
                skip = features[i]  # 对应层的编码器特征

                x = up_block(x, skip)

        # 最终上采样到目标尺寸
        x = self.final_upsample(x)

        return x


class UpBlock(nn.Module):
    """上采样块"""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
        )
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, 3, padding=1),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(
        self, x: torch.Tensor, skip: torch.Tensor | None = None
    ) -> torch.Tensor:
        x = self.up(x)

        if skip is not None:
            # 调整skip connection的尺寸（如果需要）
            if x.shape[-2:] != skip.shape[-2:]:
                skip = F.interpolate(
                    skip, size=x.shape[-2:], mode="bilinear", align_corners=True
                )

            x = torch.cat([x, skip], dim=1)

        x = self.conv(x)
        return x


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
        in_channels: int = 1,
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

    def __init__(self, in_channels=1, feature_dims: list[int] | None = None):
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
        decoded_feature_dim: int = 32,
    ):
        super().__init__()
        self.config = config
        self.encoder_type = encoder_type

        if feature_dims is None:
            if encoder_type == "swin":
                feature_dims = [64, 128, 256, 512]
            else:
                feature_dims = [64, 128, 256, 512, 1024]

        self.feature_dims = feature_dims

        # 为每个投影面创建独立的编码器
        self.view_encoders = nn.ModuleDict()
        self.view_decoders = nn.ModuleDict()
        self.view_feature_maps = nn.ModuleDict()

        for view_name, original_size in config.projection_sizes.items():
            # 编码器
            if encoder_type == "swin":
                encoder = SwinEncoder(
                    in_channels=config.projection_channels, feature_dims=feature_dims
                )
            else:
                encoder = UNetEncoder(
                    in_channels=config.projection_channels, feature_dims=feature_dims
                )
            self.view_encoders[view_name] = encoder

            # 解码器（还原到原始投影图尺寸）
            decoder = UNetLikeDecoder(
                encoder_channels=encoder.encoder_channels,
                output_channels=decoded_feature_dim,
                output_size=original_size,  # 还原到原始尺寸
            )
            self.view_decoders[view_name] = decoder

            # 最终特征映射（可选，进一步提炼特征）
            self.view_feature_maps[view_name] = nn.Sequential(
                nn.Conv2d(decoded_feature_dim, decoded_feature_dim, 3, padding=1),
                nn.BatchNorm2d(decoded_feature_dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(decoded_feature_dim, decoded_feature_dim, 1),
            )
            self.decoded_feature_dim = decoded_feature_dim

    def forward(
        self, projections: dict[str, torch.Tensor]
    ) -> dict[str, dict[str, torch.Tensor]]:
        """返回解码到原始尺寸的特征图"""
        decoded_features = {}
        encoder_features_dict = {}

        # 处理每个投影面
        for view_name, proj_tensor in projections.items():
            # 编码
            # 加通道
            proj_tensor = proj_tensor.unsqueeze(1)
            encoder_features = self.view_encoders[view_name](proj_tensor)
            encoder_features_dict[view_name] = encoder_features

            # 解码回原始尺寸
            decoded_feature = self.view_decoders[view_name](encoder_features)

            # 最终特征映射
            # final_feature = self.view_feature_maps[view_name](decoded_feature)
            # decoded_features[view_name] = final_feature
            decoded_features[view_name] = decoded_feature

        return {
            "features": decoded_features,  # 原始尺寸的特征图
            "encoder_features": encoder_features_dict,  # 编码器中间特征（可选）
        }


class PointFeatureSampler(nn.Module):
    """
    点特征采样器：从各个视图的特征图中采样点特征
    """

    def __init__(self):
        super().__init__()

    def sample_view_features(
        self,
        features: torch.Tensor,
        points: torch.Tensor,
        view_name: str,
    ):
        # 存储所有尺度的点特征
        # scale_point_features = []

        # 变换特征图
        # 将3D点投影到当前视图
        proj_coords = self.project_points_to_view(points, view_name)

        # print("66666", proj_coords.shape)

        # 双线性插值采样特征
        point_feat = index_2d(features, proj_coords)  # B C N

        point_feat = point_feat.transpose(1, 2)  # [B, N, C]
        # scale_point_features.append(point_feat)

        # 拼接所有尺度的特征
        # view_point_features = torch.cat(
        # scale_point_features, dim=-1
        # )  # [B, N, total_output_dim]

        return point_feat

    def project_points_to_view(
        self,
        points: torch.Tensor,
        view_name: str,
    ):
        """将3D点投影到2D视图坐标"""

        # 根据视图名称选择投影方式
        if view_name == "d1":
            proj_coords = points[:, :, :2].clone()  # [B, N, 2]
        elif view_name == "d2":
            proj_coords = points[:, :, :2].clone()  # [B, N, 2]
        elif view_name == "d3":
            proj_coords = points[:, :, 1:].clone()  # [B, N, 2]
        elif view_name == "d4":
            proj_coords = points[:, :, 1:].clone()  # [B, N, 2]
        else:
            raise ValueError(f"Unknown view name: {view_name}")

        # 坐标转换

        proj_coords[:, :, 0] = 2.0 * proj_coords[:, :, 0] - 1.0
        proj_coords[:, :, 1] = 2.0 * proj_coords[:, :, 1] - 1.0

        return proj_coords
