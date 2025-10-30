import torch
import numpy as np


def calc_combine_loss(loss, x, y, k=1.0):
    loss["loss_combine"] = combined_loss(x, y)
    loss["loss"] += k * loss["loss_combine"]
    return loss


def calc_mse_loss(loss, x, y, k=1.0):
    """
    Calculate mse loss.
    """
    # Compute loss
    loss_mse = torch.mean((x - y) ** 2)
    loss["loss"] += k * loss_mse
    loss["loss_mse"] = loss_mse
    return loss


def calc_mse_loss_raw(loss, x, y, k=1):
    """
    Calculate mse loss for raw.
    """
    # Compute loss for raw
    loss_mse_raw = torch.mean((x - y) ** 2)
    loss["loss"] += k * loss_mse_raw
    loss["loss_mse_raw"] = loss_mse_raw
    return loss


def calc_tv_loss(loss, x, k):
    """
    Calculate total variation loss.
    Args:
        x (n1, n2, n3, 1): 3d density field.
        k: relative weight
    """
    n1, n2, n3 = x.shape
    tv_1 = torch.abs(x[1:, 1:, 1:] - x[:-1, 1:, 1:]).sum().type(torch.float32)
    tv_2 = torch.abs(x[1:, 1:, 1:] - x[1:, :-1, 1:]).sum().type(torch.float32)
    tv_3 = torch.abs(x[1:, 1:, 1:] - x[1:, 1:, :-1]).sum().type(torch.float32)
    tv = (tv_1 + tv_2 + tv_3) / n1 / n2 / n3
    loss["loss"] += tv * k
    loss["loss_tv"] = tv * k
    print("TV loss is inf", tv_1, tv_2, tv_3, n1, n2, n3, tv)
    return loss


def calc_tv_2d_loss(loss, x, k):
    """
    Anisotropic TV loss similar to the one in [1]_.

    Parameters
    ----------
    x : :class:`torch.Tensor`
        Tensor of which to compute the anisotropic TV w.r.t. its last two axes.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Total_variation_denoising
    """
    dh = torch.abs(x[..., :, 1:] - x[..., :, :-1])
    dw = torch.abs(x[..., 1:, :] - x[..., :-1, :])
    tv = torch.sum(dh[..., :-1, :] + dw[..., :, :-1])
    loss["loss"] += tv * k
    loss["loss_tv"] = tv * k
    return loss


def compute_tv_norm(
    values, losstype="l2", weighting=None
):  # pylint: disable=g-doc-args
    """Returns TV norm for input values.

    Note: The weighting / masking term was necessary to avoid degenerate
    solutions on GPU; only observed on individual DTU scenes.
    """
    v00 = values[:, :-1, :-1]
    v01 = values[:, :-1, 1:]
    v10 = values[:, 1:, :-1]

    if losstype == "l2":
        loss = ((v00 - v01) ** 2) + ((v00 - v10) ** 2)
    elif losstype == "l1":
        loss = np.abs(v00 - v01) + np.abs(v00 - v10)
    else:
        raise ValueError("Not supported losstype.")

    if weighting is not None:
        loss = loss * weighting
    return loss


import torch
import torch.nn as nn
import torch.nn.functional as F


class SparseLightLoss(nn.Module):
    """针对稀疏光源的损失函数（抑制背景，增强光源区域权重）"""

    def __init__(self, pos_weight=150.0, sparse_weight=0.01, attn_weight=0.1):
        super().__init__()
        self.pos_weight = pos_weight  # 正样本（光源）权重
        self.sparse_weight = sparse_weight  # 稀疏性约束权重
        self.attn_weight = attn_weight  # 注意力约束权重

    # def forward(self, pred_density, gt_density, attn_weights):
    def forward(self, pred_density, gt_density):
        """
        pred_density: [B, N, 1] 预测密度
        gt_density: [B, N, 1] 真实密度（0=背景，1=光源）
        attn_weights: [B, N, num_views] 注意力权重
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

        # 3. 注意力熵约束（光源点的注意力应更集中，熵更小）
        # attn_entropy_loss = 0.0
        # light_mask = (gt_density > 0.5).squeeze(-1)  # [B, N]，光源点掩码
        # if light_mask.sum() > 0:
        #     # 仅对光源点计算注意力熵
        #     light_attn = attn_weights[light_mask]  # [K, num_views]，K为光源点数量
        #     entropy = -torch.sum(
        #         light_attn * torch.log(light_attn + 1e-8), dim=1
        #     ).mean()
        #     attn_entropy_loss = self.attn_weight * entropy

        # total_loss = bce_loss + sparse_loss + attn_entropy_loss
        total_loss = bce_loss + sparse_loss
        # total_loss = bce_loss
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
