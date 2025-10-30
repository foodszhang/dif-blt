import torch
import torch.nn as nn
import torch.nn.functional as F
from timm import create_model
import numpy as np
from .base import ProjectionConfig
from src.utils.utils import index_2d
from src.utils.cam import project_points_to_camera


class PointFeatureSampler:
    """
    点特征采样器：从各个视图的特征图中采样点特征
    """

    def project_points_to_view(
        self,
        points: torch.Tensor,
        view_name: str,
        camera_distance: int,
        detector_size: tuple[int],
        voxel_shape: tuple[int],
    ):
        """将3D点投影到2D视图坐标, 并且归一化到(-1,1)"""

        # 坐标转换
        voxel_shape = torch.asarray(voxel_shape, device=points.device)
        points = points * (voxel_shape - 1)
        # test_point = torch.asarray([[[140, 110, 150]]], device=points.device)
        # print("44444", test_point.shape)
        # test_point = test_point - voxel_shape / 2.0 + 0.5
        points = points - voxel_shape / 2.0 + 0.5
        proj, dep = project_points_to_camera(
            points, int(view_name), camera_distance, detector_size
        )
        # test_proj, _ = project_points_to_camera(
        #     test_point, int(view_name), camera_distance, detector_size
        # )
        # print(f"555555 view_{view_name}", test_proj)

        proj_coords = proj / torch.asarray(detector_size, device=points.device) * 2

        return proj_coords


class DoubleConv(nn.Module):
    """U-Net基础模块：两次卷积+BN+ReLU（官方标准实现）"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """U-Net下采样模块：MaxPool+DoubleConv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2), DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """U-Net上采样模块：双线性插值/转置卷积+特征拼接+DoubleConv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        # 双线性插值（无参数，计算量小）或转置卷积（有参数，可能更精确）
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels // 2, in_channels // 2, kernel_size=2, stride=2
            )
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        # x1: 上采样输入（低分辨率）；x2: 跳跃连接输入（高分辨率）
        x1 = self.up(x1)
        # 处理尺寸不匹配（边缘对齐）
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)  # 通道维度拼接
        return self.conv(x)


class OutConv(nn.Module):
    """U-Net输出卷积：调整通道数"""

    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    """标准U-Net实现（支持多尺度特征输出，用于特征提取）"""

    def __init__(self, n_channels, n_features=64, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_features = n_features  # 基础通道数
        self.bilinear = bilinear

        # 编码器（下采样）
        self.inc = DoubleConv(n_channels, n_features)
        self.down1 = Down(n_features, n_features * 2)
        self.down2 = Down(n_features * 2, n_features * 4)
        self.down3 = Down(n_features * 4, n_features * 8)
        factor = 2 if bilinear else 1
        self.down4 = Down(n_features * 8, n_features * 16 // factor)

        # 解码器（上采样）
        self.up1 = Up(n_features * 16, n_features * 8 // factor, bilinear)
        self.up2 = Up(n_features * 8, n_features * 4 // factor, bilinear)
        self.up3 = Up(n_features * 4, n_features * 2 // factor, bilinear)
        self.up4 = Up(n_features * 2, n_features, bilinear)

        # 多尺度特征融合（保留输入分辨率的特征图）
        self.feature_fusion = nn.Sequential(
            DoubleConv(n_features, n_features),
            OutConv(n_features, n_features),  # 输出与输入通道数一致（便于后续融合）
        )

    def forward(self, x):
        """输入单视图投影图，输出同分辨率特征图（多尺度融合）"""
        # 编码器特征
        x1 = self.inc(x)  # [B, 64, H, W]
        x2 = self.down1(x1)  # [B, 128, H/2, W/2]
        x3 = self.down2(x2)  # [B, 256, H/4, W/4]
        x4 = self.down3(x3)  # [B, 512, H/8, W/8]
        x5 = self.down4(x4)  # [B, 512, H/16, W/16]（若bilinear=True）

        # 解码器特征
        x = self.up1(x5, x4)  # [B, 256, H/8, W/8]
        x = self.up2(x, x3)  # [B, 128, H/4, W/4]
        x = self.up3(x, x2)  # [B, 64, H/2, W/2]
        x = self.up4(x, x1)  # [B, 64, H, W]

        # 融合并输出与输入同分辨率的特征图
        feature_map = self.feature_fusion(x)  # [B, 64, H, W]
        return feature_map


class PositionalEncoding3D(nn.Module):
    """3D位置编码（增强空间位置区分性，官方Transformer风格实现）"""

    def __init__(self, d_model, max_coords=1):
        super().__init__()
        self.d_model = d_model
        self.max_coords = max_coords  # 坐标最大值（用于归一化）
        assert d_model % 6 == 0, "d_model must be divisible by 6 (3 dims × sin+cos)"
        self.num_freqs = d_model // 6

    def forward(self, x):
        """
        x: [B, N, 3] 3D点坐标（x, y, z）
        输出: [B, N, d_model] 位置编码
        """
        x = x / self.max_coords * 2 - 1  # 归一化到[-1, 1]（假设坐标范围已知）
        B, N, _ = x.shape
        freqs = torch.linspace(1.0, 10.0, self.num_freqs, device=x.device)  # 频率递增
        freqs = 2 * np.pi * freqs  # 角频率

        encodings = []
        for i in range(3):  # 对x, y, z分别编码
            coord = x[..., i]  # [B, N]
            for freq in freqs:
                encodings.append(torch.sin(coord * freq))  # sin编码
                encodings.append(torch.cos(coord * freq))  # cos编码
        # 拼接所有编码
        pos_enc = torch.stack(encodings, dim=-1)  # [B, N, d_model]
        return pos_enc


class SpatialAttentionFusion(nn.Module):
    """基于3D点投影的空间注意力融合模块（不使用平均池化，特征维度不变）"""

    def __init__(self, feature_dim=64, pos_enc_dim=120, kernel_size=9):
        super().__init__()
        self.kernel_size = kernel_size  # 局部邻域大小（3x3）
        self.pos_encoder = PositionalEncoding3D(d_model=pos_enc_dim)
        self.sampler = PointFeatureSampler()

        # 注意力权重计算网络（输入：局部特征+位置编码）
        self.attention = nn.Sequential(
            # nn.Linear(feature_dim * kernel_size * kernel_size + pos_enc_dim, 256),
            # nn.Linear(feature_dim + pos_enc_dim, 256),
            nn.Linear(pos_enc_dim + 1 + feature_dim, 256),
            # nn.Linear(pos_enc_dim + 1, 256),
            # nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            # nn.BatchNorm1d(128),
            # nn.ReLU(inplace=True),
            nn.Linear(128, 1),  # 单视图权重
        )

    def forward(self, x3d, view_features):
        """
        x3d: [B, N, 3] 输入3D点坐标
        view_features: 字典，每个元素为[B, C, H_i, W_i]（不同视图的U-Net特征图）
        输出:
            fused_features: [B, N, C] 融合后的3D点特征（C=feature_dim）
            attention_weights: [B, N, num_views] 各视图注意力权重
        """
        B, N, _ = x3d.shape
        # C = list(view_features.values())[0].shape[1]  # 特征通道数
        pos_enc = self.pos_encoder(x3d)  # [B, N, pos_enc_dim]

        # 收集每个视图的局部特征和注意力分数
        local_features_list = []
        attn_logits_list = []
        view_list = ["-90", "-60", "-30", "0", "30", "60", "90"]

        for view_name in view_list:
            # 1. 获取当前视图的特征图和投影矩阵
            feat_map = view_features[view_name]
            H, W = feat_map.shape[2], feat_map.shape[3]

            # 2. 计算3D点在当前视图的2D投影坐标（平行
            # TODO: HARD CODE ,强行指定
            grid = self.sampler.project_points_to_view(
                x3d, view_name, 200, (256, 256), (182, 164, 210)
            )

            grid = grid.unsqueeze(1)  # [B, 1, N, 2]（grid_sample要求的形状）
            # pad = self.kernel_size // 2

            # padded_feat = F.pad(
            #     feat_map, (pad, pad, pad, pad), mode="replicate"
            # )  # 边缘填充
            # TODO: HARD CODE
            d_index = int(view_name)

            d_enc = torch.tensor(
                d_index, dtype=torch.float32, device=x3d.device
            ).expand(B, N, 1)
            # feat_map = feat_map.transpose(2, 3)
            local_feat = F.grid_sample(
                feat_map,
                grid,
                mode="bilinear",
                padding_mode="border",
                align_corners=True,
            )  # [B, C, 1, N]
            local_feat = local_feat.squeeze(2).permute(0, 2, 1)  # [B, N, C]

            # pad = self.kernel_size // 2
            # padded_feat = F.pad(
            #     feat_map, (pad, pad, pad, pad), mode="replicate"
            # )  # 边缘填充投影）

            # 5. 扩展为3x3邻域特征（通过滑动窗口展开，保留空间结构）
            # 方法：使用卷积核提取邻域后展平（等价于多位置采样）
            # 注：此处为高效实现，实际等价于对每个投影点采样3x3区域
            # kernel = torch.eye(self.kernel_size**2, device=feat_map.device).view(
            #     self.kernel_size**2, 1, self.kernel_size, self.kernel_size
            # )  # 生成单位卷积核，用于提取邻域
            # feat_unfold = F.conv2d(
            #     padded_feat,
            #     kernel.repeat(C, 1, 1, 1),  # [C*9, C, 3, 3]
            #     groups=C,  # 分组卷积，每个通道独立提取邻域
            # )  # [B, C*9, H, W]
            # 对展开的特征再次采样，得到每个3D点对应的3x3邻域
            # local_feat_3x3 = F.grid_sample(
            #     feat_unfold,
            #     grid,
            #     mode="bilinear",
            #     padding_mode="zeros",
            #     align_corners=True,
            # )  # [B, C*9, 1, N]
            # local_feat_3x3 = local_feat_3x3.squeeze(2).permute(0, 2, 1)  # [B, N, C*9]

            # # 6. 计算当前视图的注意力分数
            # # 拼接局部特征和位置编码
            #
            # cat_feat = torch.cat(
            #     [pos_enc, d_enc, local_feat], dim=-1
            # )  # [B, N, C*9 + pos_enc_dim]
            # # cat_feat = torch.cat([pos_enc, d_enc], dim=-1)
            # # 注意力网络输入需展平为[B*N, ...]（BatchNorm1d要求）
            # attn_input = cat_feat.reshape(B * N, -1)
            # attn_logits = self.attention(attn_input).view(B, N, 1)  # [B, N, 1]
            # attn_logits_list.append(attn_logits)
            local_features_list.append(local_feat)  # 保留原始通道特征（用于融合）

        # 7. 注意力权重归一化
        # attn_logits = torch.cat(attn_logits_list, dim=-1)  # [B, N, num_views]
        # attention_weights = F.softmax(attn_logits, dim=-1)  # [B, N, num_views]
        #
        # # 8. 加权融合特征（不使用平均池化，特征维度保持C）
        local_features = torch.cat(local_features_list, dim=-1)  # [B, N, C * num_views]
        # fused_features = torch.sum(
        #     local_features * attention_weights.unsqueeze(-2), dim=-1
        # )  # [B, N, C]

        # return fused_features, attention_weights, pos_enc
        return local_features
