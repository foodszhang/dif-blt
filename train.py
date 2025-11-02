import os
import os.path as osp
import torch
import numpy as np
import argparse
from torch.amp import autocast, GradScaler


# torch.autograd.set_detect_anomaly(True)

import torch
import gc

# 强制清理缓存


def config_parser():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--config", default=f"./config/nerf/chest_50.yaml", help="configs file path")
    parser.add_argument(
        "--config",
        default=f"./config/default.yml",
        help="configs file path",
    )
    parser.add_argument("--gpu_id", default="0", help="gpu to use")
    return parser


parser = config_parser()
args = parser.parse_args()
print("!!!!!!", torch.cuda.is_available())

# os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

from src.configloading import load_config
from src.trainer import Trainer
from src.loss import calc_mse_loss, calc_combine_loss, SparseLightLoss, dice_coefficient
from src.utils.utils import get_psnr, get_ssim, get_psnr_3d, get_ssim_3d, cast_to_image


cfg = load_config(args.config)


# torch.cuda.set_device(2)

# stx()
device = torch.device("cuda")
# stx()


# 从Trainer继承
class BasicTrainer(Trainer):
    def __init__(self):
        """
        Basic network trainer.
        """
        super().__init__(cfg, device)
        self.loss_func = SparseLightLoss(pos_weight=500, sparse_weight=0.05)
        torch.cuda.empty_cache()
        gc.collect()

        # 检查清理后的内存状态
        if torch.cuda.is_available():
            print(f"清理后空闲内存: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        print(f"[Start] exp: {cfg['exp']['expname']}, net: Basic network")

    def compute_loss(self, data, global_step, idx_epoch):
        # stx()
        loss = {"loss": 0.0}
        # stx()
        projections = data["projections"]
        points = data["points"]
        density = data["point_densities"].unsqueeze(-1)
        points = points.cuda()

        projections = {k: v.cuda() for k, v in projections.items()}
        density = density.cuda()
        # print('555555', density, density_pred)
        # with autocast("cuda"):
        density_pred, _ = self.net(projections, points)
        # if global_step % 10 == 0:
        #     print(
        #         torch.count_nonzero(density > 0.5),
        #         torch.count_nonzero(density_pred > 0.5),
        #     )

        loss = self.loss_func(density_pred, density)
        # loss += 10 * torch.mean((density - density_pred) ** 2)
        # print("44444", density_pred.max(), density_pred.min())

        return loss

    def eval_step(self, global_step, idx_epoch):
        """
        Evaluation step
        """
        # Evaluate projection    渲染投射的 RGB 图
        loss = {
            "total_dice": 0.0,
        }
        for index, data in enumerate(self.eval_dloader):
            # stx()

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
            sample_num = 40000
            points = points.cuda()

            projections = {k: v.cuda() for k, v in projections.items()}
            density = density.cuda()

            n_batch = int(np.ceil(total_npoint / sample_num))

            pred_list = []
            for i in range(n_batch):
                left = i * sample_num
                right = min((i + 1) * sample_num, total_npoint)

                pred, attn_ = self.net(projections, points[:, left:right, :])
                pred_list.append(pred)

            density_pred = torch.cat(pred_list, dim=1)
            density_pred = density_pred.reshape(voxel_shape)

            dice = dice_coefficient(density_pred, density)

            loss["total_dice"] += dice

        if loss["total_dice"] / len(self.eval_dloader) > self.best_dice:
            torch.save(
                {
                    "epoch": idx_epoch,
                    "network": self.net.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                },
                self.ckpt_best_dir,
            )  # 此处并没有save best的操作呀
            self.best_dice = loss["total_dice"] / len(self.eval_dloader)
            self.logger.info(
                f"best model update, epoch:{idx_epoch}, best 3d avg dice:{self.best_dice:.4g}"
            )
        return loss


if __name__ == "__main__":
    trainer = BasicTrainer()
    # 这并不是多线程中的start函数，而是父类Trainer中的start函数
    trainer.start()  # loop train and evaluation
