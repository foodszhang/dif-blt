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

    def __init__(self, data_dir, npoints=20000, device="cuda"):
        super().__init__()
        self.data_dir = data_dir

        # with open(os.path.join(data_dir, "info.json"), "r") as f:
        #    cfg = json.load(f)
        self.npoints = npoints
        self.type = type
        self.device = device
        voxel_path = os.path.join(data_dir, "volume_brain.npy")

        self.volume_voxels = torch.tensor(
            np.load(voxel_path), dtype=torch.float32, device=device
        )
        self.use_importance_sampling = False

        voxel_shape = self.volume_voxels.shape
        points = np.mgrid[: voxel_shape[0], : voxel_shape[1], : voxel_shape[2]]
        # points = points.astype(float) / (256 - 1)
        points = points.reshape(3, -1)
        self.points = points.transpose(1, 0)  # N, 3
        # self.points = torch.tensor(self.points, dtype=torch.int, device=device)
        entries = os.listdir(data_dir)
        self.voxel_shape = voxel_shape

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

        ##TODO: 硬编码一下
        self.proj_size = {
            "d1": (180, 300),
            "d2": (180, 300),
            "d3": (300, 208),
            "d4": (300, 208),
        }

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
        source_data = source_data.reshape(json_file["Optode"]["Source"]["Param1"])
        source_in_vol = np.zeros(self.volume_voxels.shape, dtype=np.float32)
        source_in_vol[
            source_pos[0] : source_pos[0] + source_data.shape[0],
            source_pos[1] : source_pos[1] + source_data.shape[1],
            source_pos[2] : source_pos[2] + source_data.shape[2],
        ] = source_data

        projection_zip = np.load(proj_path)
        params = ["d1", "d2", "d3", "d4"]
        projections = {
            p: torch.tensor(
                projection_zip[p] / 1e4, dtype=torch.float32, device=self.device
            )
            for p in params
        }
        points, point_densities = self.sample_points(self.points, source_in_vol)

        for i in range(3):
            points[:, i] = points[:, i] / self.voxel_shape[i]
        points = torch.tensor(points, dtype=torch.float32, device=self.device)

        # source_in_vol = torch.tensor(
        #     source_in_vol, dtype=torch.float32, device=self.device
        # )
        point_densities = torch.tensor(
            point_densities, dtype=torch.float32, device=self.device
        )

        return {
            "projections": projections,
            "points": points,
            "point_densities": point_densities,
            # "densities": source_in_vol,
        }

    def sample_points(self, points, values):
        flat_values = values.reshape(-1)
        sum_values = flat_values.sum()
        p = (flat_values + 0.1) / (sum_values + len(flat_values) * 0.1)

        choice = np.random.choice(len(points), size=self.npoints, replace=False, p=p)

        points = points[choice]

        if values is not None:
            value = values[
                points[:, 0].astype(int),
                points[:, 1].astype(int),
                points[:, 2].astype(int),
            ]
            # values = values.astype(float)
            return points, value
        else:
            return points
