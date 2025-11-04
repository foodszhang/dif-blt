import torch
import os
import numpy as np
import json

from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from scipy.ndimage import zoom


# dataloader，把数据做成 TIGRE 数据类型
class MultiProjDataset(Dataset):
    """
    多视图投影数据集
    """

    def __init__(self, data_dir, block_dir, device="cpu", is_training=True):
        super().__init__()
        self.data_dir = data_dir
        self.block_dir = block_dir
        # 获取所有块文件路径
        self.block_files = [
            os.path.join(block_dir, f)
            for f in os.listdir(block_dir)
            if f.endswith(".npz")
        ]
        self.block_files.sort(
            key=lambda x: int(os.path.basename(x).split("_")[1].split(".")[0])
        )
        self.max_voxels = 64 * 64 * 64

        # with open(os.path.join(data_dir, "info.json"), "r") as f:
        #    cfg = json.load(f)
        # self.npoints = npoints
        self.type = type
        self.device = device
        voxel_path = os.path.join(data_dir, "volume_brain.npy")
        volume_voxel = np.load(voxel_path)
        self.origin_voxel_shape = volume_voxel.shape
        # volume_voxel = zoom(volume_voxel, 0.5)
        # volume_voxel = np.where(volume_voxel > 0.5, 1, 0)
        # self.volume_voxels = torch.tensor(
        #     volume_voxel, dtype=torch.float32, device=device
        # )

        self.use_importance_sampling = False

        voxel_shape = volume_voxel.shape
        # TODO: HARD CODE
        # range_x = (60, 120)
        # range_y = (40, 140)
        # range_z = (96, 130)
        range_x = (40, 120)
        range_y = (20, 140)
        range_z = (80, 160)

        points = np.mgrid[
            range_x[0] : range_x[1],
            range_y[0] : range_y[1],
            range_z[0] : range_z[1],
        ]

        # points = np.mgrid[: voxel_shape[0], : voxel_shape[1], : voxel_shape[2]]
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

        self.is_training = is_training

    def __len__(self):
        return self.total_num
        # return 800
        # if self.is_training:
        #     return 20
        # else:
        #     return 4

    def __getitem__(self, index):
        block_idx = np.random.randint(len(self.block_files))
        block_path = self.block_files[block_idx]
        data = np.load(block_path)

        # 加载坐标和值并转换为张量
        # coords = torch.from_numpy(data["coords"]).float()  # (N, 3)
        coords = data["coords"]
        entry, entry_path = self.dirs[index]
        proj_path = os.path.join(entry_path, "proj.npz")
        # proj_path = os.path.join(entry_path, "no_proj.npz")
        proj_no_path = os.path.join(entry_path, "no_proj.npz")
        json_path = os.path.join(entry_path, f"{entry}.json")
        json_file = json.load(open(json_path))
        source_pos = json_file["Optode"]["Source"]["Pos"]
        source_pattern = json_file["Optode"]["Source"]["Pattern"]
        source_data = np.fromfile(
            os.path.join(entry_path, source_pattern["Data"]), dtype=np.float32
        )
        source_shape = json_file["Optode"]["Source"]["Param1"]
        source_data = source_data.reshape(
            source_shape[2], source_shape[1], source_shape[0]
        )
        source_data = source_data.transpose(2, 1, 0)
        source_in_vol = np.zeros(self.origin_voxel_shape, dtype=np.float32)
        source_in_vol[
            source_pos[0] : source_pos[0] + source_data.shape[0],
            source_pos[1] : source_pos[1] + source_data.shape[1],
            source_pos[2] : source_pos[2] + source_data.shape[2],
        ] = source_data
        # source_in_vol = zoom(source_in_vol, 0.5)
        # source_in_vol = np.where(source_in_vol > 0.5, 1, 0)

        projection_zip = np.load(proj_path)
        no_projection_zip = np.load(proj_no_path)
        # TODO: HARD CODE
        params = ["-90", "-60", "-30", "0", "30", "60", "90"]
        projections = {
            p: torch.tensor(
                projection_zip[p] / 1e5,
                dtype=torch.float32,
                device=self.device,
            )
            for p in params
        }
        no_projections = {
            p: torch.tensor(
                no_projection_zip[p] / 1e5,
                dtype=torch.float32,
                device=self.device,
            )
            for p in params
        }
        if self.is_training:
            N = coords.shape[0]
            pad_size = self.max_voxels - N
            coords = np.pad(coords, ((0, pad_size), (0, 0)), mode="constant")
            points = coords
            point_densities = source_in_vol[
                points[:, 0].astype(int),
                points[:, 1].astype(int),
                points[:, 2].astype(int),
            ]

            points = torch.tensor(points, dtype=torch.float32, device=self.device)
            points = points / (
                torch.asarray(self.voxel_shape, device=self.device, dtype=torch.float32)
                - 1
            )
            # source_in_vol = torch.tensor(
            #     source_in_vol, dtype=torch.float32, device=self.device
            # )
            point_densities = torch.tensor(
                point_densities, dtype=torch.float32, device=self.device
            )

            return {
                "projections": projections,
                "no_projections": no_projections,
                "points": points,
                "point_densities": point_densities,
                "total_num": np.count_nonzero(source_in_vol > 0.1),
                "file_index": index,
                # "densities": source_in_vol,
            }
        else:
            point_densities = source_in_vol[
                self.points[:, 0].astype(int),
                self.points[:, 1].astype(int),
                self.points[:, 2].astype(int),
            ]
            point_densities = torch.tensor(
                point_densities, dtype=torch.float32, device=self.device
            )
            points = torch.tensor(self.points, dtype=torch.float32, device=self.device)

            points = points / (
                torch.asarray(self.voxel_shape, device=self.device, dtype=torch.float32)
                - 1
            )
            return {
                "projections": projections,
                "no_projections": no_projections,
                "points": points,
                "point_densities": point_densities,
            }

    def sample_points(self, points, values):
        flat_values = values[
            points[:, 0].astype(int),
            points[:, 1].astype(int),
            points[:, 2].astype(int),
        ]
        sum_values = flat_values.sum()
        p = (flat_values + 0.05) / (sum_values + len(flat_values) * 0.05)

        # choice = np.random.choice(len(points), size=self.npoints, replace=False, p=p)
        choice = np.random.choice(len(points), size=self.npoints, replace=False)

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
