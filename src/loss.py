import torch
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim


class ScatterLightLoss(nn.Module):
    def __init__(
        self,
        # 散射校正损失参数
        init_scatter_weight=1.0,  # 初始权重（训练初期）
        target_scatter_weight=0.5,  # 目标权重（训练后期）
        start_decay_epoch=200,  # 开始衰减的epoch
        decay_epochs=100,  # 衰减持续epoch（200→300逐步衰减）
        # SparseLightLoss参数
        pos_weight=150.0,
        sparse_weight=0.01,
    ):
        super().__init__()
        self.init_scatter_weight = init_scatter_weight
        self.target_scatter_weight = target_scatter_weight
        self.start_decay_epoch = start_decay_epoch
        self.decay_epochs = decay_epochs
        self.current_epoch = 0  # 需外部传入当前epoch

        # 初始化子损失
        self.sparse_light_loss = SparseLightLoss(pos_weight, sparse_weight)
        self.l1_loss = nn.L1Loss()

    def update_epoch(self, epoch):
        """训练循环中调用，更新当前epoch（用于权重调度）"""
        self.current_epoch = epoch

    def get_dynamic_scatter_weight(self):
        """根据当前epoch计算动态散射校正权重"""
        if self.current_epoch < self.start_decay_epoch:
            # 训练初期：保持初始权重（优先校正）
            return self.init_scatter_weight
        elif self.current_epoch < self.start_decay_epoch + self.decay_epochs:
            # 衰减阶段：线性降低权重
            decay_ratio = (
                self.current_epoch - self.start_decay_epoch
            ) / self.decay_epochs
            return self.init_scatter_weight - decay_ratio * (
                self.init_scatter_weight - self.target_scatter_weight
            )
        else:
            # 训练后期：保持目标权重（优先光源预测）
            return self.target_scatter_weight

    def scatter_correction_loss(self, pred_scatter: dict, gt_scatter: dict):
        """散射校正损失（L1+SSIM）"""
        total_scatter_loss = 0.0
        for angle in pred_scatter.keys():
            pred = pred_scatter[angle]
            gt = gt_scatter[angle]

            l1 = self.l1_loss(pred, gt)

            # SSIM损失计算
            pred_np = pred.detach().cpu().squeeze().numpy()
            gt_np = gt.detach().cpu().squeeze().numpy()
            ssim_total = 0.0
            for b in range(pred_np.shape[0]):
                ssim_val = ssim(pred_np[b], gt_np[b], data_range=1.0)
                ssim_total += 1 - ssim_val
            ssim_loss = torch.tensor(ssim_total / pred_np.shape[0], device=pred.device)

            total_scatter_loss += l1 + ssim_loss
        return total_scatter_loss / len(pred_scatter)

    def forward(self, pred_scatter, gt_scatter, pred_density, gt_density):
        # 1. 获取动态权重
        scatter_weight = self.get_dynamic_scatter_weight()

        # 2. 计算各部分损失
        scatter_loss = self.scatter_correction_loss(pred_scatter, gt_scatter)
        light_loss = self.sparse_light_loss(pred_density, gt_density)

        # 3. 加权组合总损失
        total_loss = scatter_weight * scatter_loss + light_loss

        # 返回损失及当前权重（便于监控）
        return {
            "total_loss": total_loss,
            "scatter_loss": scatter_loss,
            "light_loss": light_loss,
            "scatter_weight": scatter_weight,  # 监控权重变化
        }


# 复用原版SparseLightLoss
class SparseLightLoss(nn.Module):
    def __init__(self, pos_weight=150.0, sparse_weight=0.01, attn_weight=0.1):
        super().__init__()
        self.pos_weight = pos_weight
        self.sparse_weight = sparse_weight
        self.attn_weight = attn_weight

    def forward(self, pred_density, gt_density):
        bce_loss = F.binary_cross_entropy(
            pred_density,
            gt_density,
            weight=gt_density * (self.pos_weight - 1) + 1,
        )
        sparse_loss = self.sparse_weight * torch.mean(torch.abs(pred_density))
        return bce_loss + sparse_loss


# 复用你的原版SparseLightLoss（未改动）
class SparseLightLoss(nn.Module):
    """针对稀疏光源的损失函数（抑制背景，增强光源区域权重）"""

    def __init__(self, pos_weight=150.0, sparse_weight=0.01, attn_weight=0.1):
        super().__init__()
        self.pos_weight = pos_weight  # 正样本（光源）权重
        self.sparse_weight = sparse_weight  # 稀疏性约束权重
        self.attn_weight = attn_weight  # 注意力约束权重

    def forward(self, pred_density, gt_density):
        """
        pred_density: [B, N, 1] 预测密度
        gt_density: [B, N, 1] 真实密度（0=背景，1=光源）
        """
        # 1. 加权二元交叉熵（提升光源区域损失权重）
        bce_loss = F.binary_cross_entropy(
            pred_density,
            gt_density,
            weight=gt_density * (self.pos_weight - 1)
            + 1,  # 正样本权重=pos_weight，负样本=1
        )

        # 2. L1稀疏性约束（抑制背景区域的预测值）
        sparse_loss = self.sparse_weight * torch.mean(torch.abs(pred_density))

        total_loss = bce_loss + sparse_loss
        return total_loss


def dice_coefficient(pred, target, threshold=0.5, eps=1e-8):
    """
    计算三维体素二分类的Dice系数

    参数:
        pred: 模型输出的预测值，shape为(B, D, H, W)，通常是sigmoid后的概率值
        target: 真实标签，shape为(B, D, H, W)，值为0或1
        threshold: 二值化阈值，默认0.5
        eps: 防止分母为0的微小值

    返回:
        批次的平均Dice系数（ scalar ）
    """
    # 1. 预测值二值化（二分类）
    pred_bin = (pred >= threshold).float()  # 转换为0/1的float类型

    # 2. 计算交（intersection）和并（union的分子部分）
    intersection = (pred_bin * target).sum(
        dim=(1, 2, 3)
    )  # 对D、H、W维度求和，保留批次维度(B,)
    pred_sum = pred_bin.sum(dim=(1, 2, 3))  # 预测正类总和 (B,)
    target_sum = target.sum(dim=(1, 2, 3))  # 真实正类总和 (B,)

    # 3. 计算每个样本的Dice系数，再求批次平均
    dice_per_sample = (2.0 * intersection + eps) / (pred_sum + target_sum + eps)
    return dice_per_sample.mean()  # 返回批次平均Dice


def compute_dice(pred_mask, gt_mask):
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = pred_mask.sum() + gt_mask.sum()
    if union == 0:
        return 1.0  # 均无光源时视为完全匹配
    return 2 * intersection / (union + 1e-8)
