import os
import os.path as osp
import torch
import numpy as np
import argparse

# torch.autograd.set_detect_anomaly(True)


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

from src.configloading import load_config
from src.trainer import Trainer
from src.loss import calc_mse_loss, calc_combine_loss
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
        print(f"[Start] exp: {cfg['exp']['expname']}, net: Basic network")

    def compute_loss(self, data, global_step, idx_epoch):
        # stx()
        loss = {"loss": 0.0}
        # stx()
        projections = data["projections"]
        points = data["points"]
        density = data["point_densities"]
        density_pred = self.net(projections, points).squeeze(-1)
        calc_mse_loss(loss, density, density_pred)
        calc_combine_loss(loss, density_pred.unsqueeze(1), density.unsqueeze(1))
        return loss["loss"]

    def eval_step(self, global_step, idx_epoch):
        """
        Evaluation step
        """
        # Evaluate projection    渲染投射的 RGB 图
        loss = {
            "psnr_3d": 0.0,
            "ssim_3d": 0.0,
        }
        for index, data in enumerate(self.eval_dloader):
            loss = {"loss": 0.0}
            # stx()
            projections = data["projections"]
            points = data["points"]
            density = data["point_densities"]
            density_pred = self.net(projections, points).squeeze(1)
            calc_mse_loss(loss, density, density_pred)
            calc_combine_loss(loss, density_pred.unsqueeze(1), density.unsqueeze(1))

        return loss


trainer = BasicTrainer()
# 这并不是多线程中的start函数，而是父类Trainer中的start函数
trainer.start()  # loop train and evaluation
