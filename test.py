import torch
import numpy as np
import os
from src.dataset.proj_dataset import MultiProjDataset as Dataset
from torch.utils.data import DataLoader
from src.loss import SparseLightLoss

from src.network import get_network

ckpt_dir = "./logs/unet/unet_density/2025_10_29_22_42_26/ckpt.tar"
# ckpt_dir = "./one_source/ckpt.tar"
device = "cuda"
model = get_network("density", 7).to(device)
ckpt = torch.load(ckpt_dir, weights_only=False)
model.load_state_dict(ckpt["network"])
model.eval()
# eval_dataset = Dataset("../mcx_simulation/one_source_val/", is_training=False)
# eval_dataset = Dataset("../mcx_simulation/one_source_train/", is_training=False)
eval_dataset = Dataset(
    "../mcx_simulation/one_source_train/",
    "../mcx_simulation/preprocessed_blocks/",
    is_training=False,
)
# eval_dataset = Dataset("../mcx_simulation/20251021/")
os.makedirs("./results", exist_ok=True)
save_dir = "./results/20251020/"
eval_dloader = DataLoader(eval_dataset, batch_size=1, shuffle=False, num_workers=0)

loss_func = SparseLightLoss(pos_weight=200, sparse_weight=0.05)


def eval_step(network, dataset):
    for index, data in enumerate(dataset):
        projections = data["projections"]
        projections = {k: v.cuda() for k, v in projections.items()}
        points = data["points"]
        B, total_npoint, _ = points.shape
        # total_npoint = points.shape[1]
        point_densities = data["point_densities"]
        # print("666666", point_densities.shape)
        voxel_shape = (B, 80, 120, 80)
        density = point_densities.reshape(voxel_shape)
        # density = point_densities.squeeze(0)
        sample_num = 10000
        points = points.cuda()

        density = density.cuda()

        n_batch = int(np.ceil(total_npoint / sample_num))

        pred_list = []
        for i in range(n_batch):
            left = i * sample_num
            right = min((i + 1) * sample_num, total_npoint)

            pred, attn_ = network(projections, points[:, left:right, :])
            pred_list.append(pred)

        density_pred = torch.cat(pred_list, dim=1)
        density_pred = density_pred.reshape(voxel_shape)
        loss = loss_func(density_pred, density)
        print("3333", torch.count_nonzero(density > 0.5))
        print("4444", torch.count_nonzero(density_pred > 0.5))
        print("5555", loss)
        # p_loss = self.loss_func(density_pred, density)

        # loss["total_loss"] += p_loss

        source_mask = (density > 0.5).type(torch.bool)
        pred_mask = (density_pred > 0.5).type(torch.bool)
        source_mask = source_mask.cpu().detach().numpy()
        pred_mask = pred_mask.cpu().detach().numpy()
        dice_loss = compute_dice(pred_mask, source_mask)
        print("!!!!!!", dice_loss)
        pred_mask.astype(np.uint8).tofile(f"./results/pred_{index}.bin")
        source_mask.astype(np.uint8).tofile(f"./results/gt_{index}.bin")
        if index > 3:
            break
        # break


def compute_dice(pred_mask, gt_mask):
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = pred_mask.sum() + gt_mask.sum()
    if union == 0:
        return 1.0  # 均无光源时视为完全匹配
    return 2 * intersection / (union + 1e-8)


with torch.no_grad():
    eval_step(model, eval_dloader)
