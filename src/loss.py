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


def compute_tv_norm(values, losstype="l2", weighting=None):  # pylint: disable=g-doc-args
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


class DiceLoss(nn.Module):
    """Dice Loss for 3D density field segmentation"""

    def __init__(self, smooth=1e-6, gamma=1.0, weight=None):
        super().__init__()
        self.smooth = smooth
        self.gamma = gamma  # 用于调整难易样本权重
        self.weight = weight  # 类别权重

    def forward(self, prediction, target):
        """
        prediction: (B, C, D, H, W) 或 (B, C, H, W) 或 (B, C, N)
        target: (B, D, H, W) 或 (B, H, W) 或 (B, N)
        """
        # 确保预测是概率
        if prediction.shape[1] > 1:
            # 多分类情况
            prediction = F.softmax(prediction, dim=1)
        else:
            # 二分类情况
            prediction = torch.sigmoid(prediction)

        # 处理不同的输入维度
        if prediction.dim() == 5:  # 3D volume
            B, C, D, H, W = prediction.shape
            prediction = prediction.view(B, C, -1)
            target = target.view(B, -1)
        elif prediction.dim() == 4:  # 2D feature
            B, C, H, W = prediction.shape
            prediction = prediction.view(B, C, -1)
            target = target.view(B, -1)
        elif prediction.dim() == 3:  # Point cloud
            B, C, N = prediction.shape
            prediction = prediction.view(B, C, -1)
            target = target.view(B, -1)

        # 计算Dice系数
        if prediction.shape[1] == 1:
            # 二分类
            intersection = (prediction * target.unsqueeze(1)).sum(dim=2)
            union = prediction.sum(dim=2) + target.sum(dim=1) + self.smooth
            dice = (2.0 * intersection + self.smooth) / union

        else:
            # 多分类
            dice = 0
            num_classes = prediction.shape[1]
            for cls in range(num_classes):
                pred_cls = prediction[:, cls, :]
                target_cls = (target == cls).float()

                intersection = (pred_cls * target_cls).sum(dim=1)
                union = pred_cls.sum(dim=1) + target_cls.sum(dim=1) + self.smooth

                if self.weight is not None:
                    class_weight = self.weight[cls]
                else:
                    class_weight = 1.0

                dice += class_weight * (2.0 * intersection + self.smooth) / union

            dice /= num_classes

        # 应用gamma调整（关注难样本）
        dice = torch.pow(dice, self.gamma)

        return 1 - dice.mean()


class CombinedLoss(nn.Module):
    """组合损失函数：Dice Loss + Focal Loss + 权重平衡"""

    def __init__(self, alpha=0.7, beta=0.3, gamma=2.0, class_weights=None):
        super().__init__()
        self.alpha = alpha  # Dice Loss权重
        self.beta = beta  # Focal Loss权重

        self.dice_loss = DiceLoss(gamma=1.0, weight=class_weights)
        self.focal_loss = FocalLoss(gamma=gamma, weight=class_weights)

    def forward(self, prediction, target):
        dice = self.dice_loss(prediction, target)
        # focal = self.focal_loss(prediction, target)

        # return self.alpha * dice + self.beta * focal
        return self.alpha * dice


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance"""

    def __init__(self, gamma=2.0, weight=None, reduction="mean"):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction

    def forward(self, prediction, target):
        if prediction.shape[1] == 1:
            # 二分类
            ce_loss = F.binary_cross_entropy_with_logits(
                prediction.squeeze(1),
                target.float(),
                weight=self.weight,
                reduction="none",
            )
            pt = torch.exp(-ce_loss)
            focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        else:
            # 多分类
            ce_loss = F.cross_entropy(
                prediction, target.long(), weight=self.weight, reduction="none"
            )
            pt = torch.exp(-ce_loss)
            focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


class_weights = torch.tensor([0.1, 0.9])  # [光源, 背景]
combined_loss = CombinedLoss(
    alpha=0.7,  # Dice Loss权重
    beta=0.3,  # Focal Loss权重
    gamma=2.0,  # Focal Loss的gamma
    class_weights=class_weights,
)


# 训练示例
def train_light_source_segmentation():
    # 网络和损失函数
    model = LightSourceSegmentationNetwork(feature_dim=256, num_classes=2)

    # 计算类别权重（根据您的数据分布调整）
    # 假设光源占比约1%，背景占比99%

    # 组合损失函数
    criterion = CombinedLoss(
        alpha=0.7,  # Dice Loss权重
        beta=0.3,  # Focal Loss权重
        gamma=2.0,  # Focal Loss的gamma
        class_weights=class_weights,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # 训练循环
    for epoch in range(num_epochs):
        for batch_idx, (
            points_3d,
            view_features,
            projection_matrices,
            targets,
        ) in enumerate(train_loader):
            optimizer.zero_grad()

            # 前向传播
            seg_logits, attention_weights = model(
                points_3d, view_features, projection_matrices
            )

            # 计算损失
            loss = criterion(seg_logits, targets)

            # 反向传播
            loss.backward()
            optimizer.step()

            # 记录和打印
            if batch_idx % 100 == 0:
                print(f"Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item():.4f}")


# 评估指标
class SegmentationMetrics:
    """分割评估指标"""

    def __init__(self, num_classes=2):
        self.num_classes = num_classes

    def compute_iou(self, prediction, target):
        """计算IoU"""
        prediction = torch.argmax(prediction, dim=1)
        ious = []

        for cls in range(self.num_classes):
            pred_cls = prediction == cls
            target_cls = target == cls

            intersection = (pred_cls & target_cls).float().sum()
            union = (pred_cls | target_cls).float().sum()

            iou = (intersection + 1e-6) / (union + 1e-6)
            ious.append(iou.item())

        return ious

    def compute_dice(self, prediction, target):
        """计算Dice系数"""
        prediction = torch.argmax(prediction, dim=1)
        dices = []

        for cls in range(self.num_classes):
            pred_cls = prediction == cls
            target_cls = target == cls

            intersection = (pred_cls & target_cls).float().sum()
            dice = (2.0 * intersection + 1e-6) / (
                pred_cls.float().sum() + target_cls.float().sum() + 1e-6
            )
            dices.append(dice.item())

        return dices
