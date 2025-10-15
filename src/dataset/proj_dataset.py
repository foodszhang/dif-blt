import torch
import os
import numpy as np
import json

from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F


# dataloader，把数据做成 TIGRE 数据类型
class MultiProjDataset(Dataset):
    """
    多视图投影数据集
    """

    def __init__(self, data_dir, device="cuda"):
        super().__init__()
        self.data_dir = data_dir

        # with open(os.path.join(data_dir, "info.json"), "r") as f:
        #    cfg = json.load(f)
        self.type = type
        self.device = device
        voxel_path = os.path.join(data_dir, "volume_bin.npy")

        self.volume_voxels = torch.tensor(
            np.load(voxel_path), dtype=torch.float32, device=device
        )
        self.use_importance_sampling = False

        voxel_shape = self.volume_voxels.shape
        points = np.mgrid[: voxel_shape[0], : voxel_shape[1], : voxel_shape[2]]
        # points = points.astype(float) / (256 - 1)
        points = points.reshape(3, -1)
        self.points = points.transpose(1, 0)  # N, 3
        self.points = torch.tensor(self.points, dtype=torch.float32, device=device)
        entries = os.listdir(data_dir)

        count = 0
        numeric_folders = []

        for entry in entries:
            entry_path = os.path.join(data_dir, entry)

            # 检查是否是文件夹
            if os.path.isdir(entry_path):
                # 检查文件夹名是否只包含数字
                if entry.isdigit():
                    count += 1
                    numeric_folders.append((entry, entry_path))
        self.total_num = count
        self.dirs = numeric_folders
        self.proj_size = None

    def __len__(self):
        return self.total_num

    def __getitem__(self, index):
        entry, entry_path = self.dirs[index]
        proj_path = os.path.join(entry_path, "proj.npz")
        json_path = os.path.join(entry_path, f"{entry}.json")
        json_file = json.load(open(json_path))
        source_pos = json_file["Optode"]["Source"]["Pos"]
        source_pattern = json_file["Optode"]["Source"]["Pattern"]
        source_data = np.fromfile(
            os.path.join(entry_path, source_pattern["Data"]), dtype=np.float32
        )
        source_data = source_data.reshape(json_file["Optode"]["Param1"])
        source_in_vol = np.zeros(self.volume_voxels.shape, dtype=np.float32)
        source_in_vol[
            source_pos[0] : source_pos[0] + source_data.shape[0],
            source_pos[1] : source_pos[1] + source_data.shape[1],
            source_pos[2] : source_pos[2] + source_data.shape[2],
        ] = source_data
        source_in_vol = torch.tensor(
            source_in_vol, dtype=torch.float32, device=self.device
        )

        projection_zip = np.load(proj_path)
        params = ["d1", "d2", "d3", "d4"]
        projections = {p: projection_zip[p] for p in params}
        projections = torch.tensor(projections, dtype=torch.float32, device=self.device)
        if self.proj_size is None:
            self.proj_size = {p: projection_zip[p].shape for p in params}
        points, point_densities = self._sample_points_from_density(source_in_vol)

        return {
            "projections": projections,
            "points": points,
            "point_densities": point_densities,
        }

    def _sample_points_from_density(
        self, density: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """从密度场中采样点和对应的密度值"""
        d, h, w = density.shape

        if self.use_importance_sampling:
            # 重要性采样：在密度高的区域多采样
            points, point_densities = self._importance_sampling(density)
        else:
            # 均匀随机采样
            points, point_densities = self._uniform_sampling(density)

        return points, point_densities

    def _uniform_sampling(
        self, density: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """均匀随机采样"""
        d, h, w = density.shape

        # 生成随机点坐标 [points_per_sample, 3]
        points = torch.rand(self.points_per_sample, 3)

        # 将坐标转换为网格索引
        grid_indices = (points * torch.tensor([w - 1, h - 1, d - 1])).long()
        x_idx, y_idx, z_idx = grid_indices[:, 0], grid_indices[:, 1], grid_indices[:, 2]

        # 获取对应点的密度值
        point_densities = density[z_idx, y_idx, x_idx].unsqueeze(
            -1
        )  # [points_per_sample, 1]

        return points, point_densities

    def _importance_sampling(
        self, density: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """重要性采样：在密度高的区域多采样"""
        d, h, w = density.shape
        total_voxels = d * h * w

        # 扁平化密度场并计算采样概率
        flat_density = density.view(-1)
        probs = F.softmax(flat_density * 10, dim=0)  # 放大差异

        # 根据概率采样点索引
        indices = torch.multinomial(probs, self.points_per_sample, replacement=True)

        # 将扁平索引转换为3D坐标
        z_idx = indices // (h * w)
        y_idx = (indices % (h * w)) // w
        x_idx = indices % w

        # 转换为归一化坐标 [0, 1]
        points = torch.stack(
            [x_idx.float() / (w - 1), y_idx.float() / (h - 1), z_idx.float() / (d - 1)],
            dim=-1,
        )  # [points_per_sample, 3]

        # 添加小噪声增加多样性
        points = points + torch.randn_like(points) * 0.01
        points = torch.clamp(points, 0, 1)

        # 获取密度值
        point_densities = flat_density[indices].unsqueeze(-1)  # [points_per_sample, 1]

        return points, point_densities
