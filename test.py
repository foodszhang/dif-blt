import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from src.dataset.proj_dataset import MultiProjDataset as Dataset
from torch.utils.data import DataLoader
from src.loss import SparseLightLoss

from src.network import get_network

# ckpt_dir = "./logs/unet/unet_density/2025_11_03_00_24_34/ckpt_best.tar"
# ckpt_dir = "./one_source/ckpt.tar"
ckpt_dir = "./one_source/ckpt_best.tar"
device = "cuda"
model = get_network("density", 7).to(device)
ckpt = torch.load(ckpt_dir, weights_only=False)
# print("5345345345", ckpt["network"].keys())
model.load_state_dict(ckpt["network"])
model.eval()
# eval_dataset = Dataset("../mcx_simulation/one_source_val/", is_training=False)
# eval_dataset = Dataset("../mcx_simulation/one_source_train/", is_training=False)
eval_dataset = Dataset(
    "../mcx_simulation/one_source_val/",
    "../mcx_simulation/preprocessed_blocks/",
    is_training=False,
)
# eval_dataset = Dataset("../mcx_simulation/20251021/")
os.makedirs("./results", exist_ok=True)
save_dir = "./results/20251020/"
eval_dloader = DataLoader(eval_dataset, batch_size=1, shuffle=False, num_workers=0)

loss_func = SparseLightLoss(pos_weight=200, sparse_weight=0.05)


def save_projection_comparison(projections, no_projections, save_dir, index):
    os.makedirs(save_dir, exist_ok=True)
    for angle, proj in projections.items():
        no_proj = no_projections[angle]

        # Convert tensors to numpy arrays
        proj_np = proj.squeeze(0).cpu().detach().numpy()
        no_proj_np = no_proj.squeeze(0).cpu().detach().numpy()

        # Create a side-by-side comparison plot
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(proj_np, cmap="hot")
        axes[0].set_title(f"Projection ({angle})")
        axes[1].imshow(no_proj_np, cmap="hot")
        axes[1].set_title(f"No Projection ({angle})")

        for ax in axes:
            ax.axis("off")

        # Save the comparison image
        plt.savefig(
            os.path.join(save_dir, f"comparison_{index}_{angle}.png"),
            bbox_inches="tight",
        )
        plt.close(fig)


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
        no_projections = {
            k: torch.zeros_like(v) for k, v in projections.items()
        }  # Temporary placeholder

        for i in range(n_batch):
            left = i * sample_num
            right = min((i + 1) * sample_num, total_npoint)

            pred, no_projections_ = network(projections, points[:, left:right, :])
            pred_list.append(pred)
            if no_projections_:
                no_projections = no_projections_  # Update the no_projections dictionary

        no_projections = data["no_projections"]
        density_pred = torch.cat(pred_list, dim=1)
        density_pred = density_pred.reshape(voxel_shape)
        loss = loss_func(density_pred, density)
        print("3333", torch.count_nonzero(density > 0.5))
        print("4444", torch.count_nonzero(density_pred > 0.5))
        print("5555", loss)

        source_mask = (density > 0.5).type(torch.bool)
        pred_mask = (density_pred > 0.5).type(torch.bool)
        source_mask = source_mask.cpu().detach().numpy()
        pred_mask = pred_mask.cpu().detach().numpy()
        dice_loss = compute_dice(pred_mask, source_mask)
        print("!!!!!!", dice_loss)

        # Save projections and no_projections as comparison images
        save_projection_comparison(
            projections, no_projections, "./results/projection_comparison", index
        )

        pred_mask.astype(np.uint8).tofile(f"./results/pred_{index}.bin")
        source_mask.astype(np.uint8).tofile(f"./results/gt_{index}.bin")


def compute_dice(pred_mask, gt_mask):
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = pred_mask.sum() + gt_mask.sum()
    if union == 0:
        return 1.0  # 均无光源时视为完全匹配
    return 2 * intersection / (union + 1e-8)


with torch.no_grad():
    eval_step(model, eval_dloader)
