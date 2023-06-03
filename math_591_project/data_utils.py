from typing import Union
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split


def teds_projection(x: Union[np.ndarray, torch.Tensor], a):
    """Projection of x onto the interval [a, a + 2*pi)"""
    return (
        np.mod(x - a, 2 * np.pi) + a
        if isinstance(x, np.ndarray)
        else torch.fmod(x - a, 2 * np.pi) + a
    )


def wrapToPi(x):
    """Wrap angles to [-pi, pi)"""
    return teds_projection(x, -np.pi)


def unwrapToPi(x: Union[np.ndarray, torch.Tensor]):
    # remove discontinuities caused by wrapToPi
    if isinstance(x, np.ndarray):
        diffs = np.diff(x)
        diffs[diffs > 1.5 * np.pi] -= 2 * np.pi
        diffs[diffs < -1.5 * np.pi] += 2 * np.pi
        return np.insert(x[0] + np.cumsum(diffs), 0, x[0])
    else:
        diffs = torch.diff(x)
        diffs[diffs > 1.5 * np.pi] -= 2 * np.pi
        diffs[diffs < -1.5 * np.pi] += 2 * np.pi
        return torch.insert(x[0] + torch.cumsum(diffs), 0, x[0])


def load_data(file_path: str, format="numpy") -> torch.Tensor:
    df = pd.read_csv(file_path)
    timestamps = df["timestamp"].to_numpy()
    x_cols = (
        ["X", "Y", "phi", "v_x"]
        + (["v_y", "r"] if "v_y" in df.columns and "r" in df.columns else [])
        + ["last_delta"]
    )
    x = df[x_cols].to_numpy()
    # make 3rd column continuous
    x[:, 2] = unwrapToPi(x[:, 2])
    Nf = int(df.columns[-1].split("_")[-1]) + 1
    x_ref_cols = []
    u_ref_cols = []
    for i in range(Nf):
        x_ref_cols.extend([f"X_ref_{i}", f"Y_ref_{i}", f"phi_ref_{i}", f"v_x_ref_{i}"])
        u_ref_cols.extend([f"T_{i}", f"ddelta_{i}"])
    x_ref_cols.extend([f"X_ref_{Nf}", f"Y_ref_{Nf}", f"phi_ref_{Nf}", f"v_x_ref_{Nf}"])
    x_ref = df[x_ref_cols].to_numpy()
    u_ref = df[u_ref_cols].to_numpy()
    if format == "torch":
        timestamps = torch.from_numpy(timestamps).to(dtype=torch.float32)
        x = torch.from_numpy(x).to(dtype=torch.float32)
        x_ref = torch.from_numpy(x_ref).to(dtype=torch.float32)
        u_ref = torch.from_numpy(u_ref).to(dtype=torch.float32)

    return timestamps, x, x_ref, u_ref


class SysidDataset(Dataset):
    xtilde0: torch.Tensor  # shape (N, 1, nx)
    utilde0toNfminus1: torch.Tensor  # shape (N, Nf, nu)
    xtilde1toNf: torch.Tensor  # shape (N, Nf, nx)
    Nf: int

    def __init__(self, file_paths: list[str], Nf: int = 1) -> None:
        super().__init__()
        assert Nf >= 1, "Nf must be greater than or equal to 1 but is {Nf}"
        xtilde0, utilde0toNfminus1, xtilde1toNf = [], [], []
        for path in file_paths:
            _, x, _, u_ref = load_data(path, format="torch")
            idx = torch.arange(0, x.shape[0] - Nf)
            xtilde0.extend([x[i : i + 1, :-1] for i in idx])
            xtilde1toNf.extend([x[i + 1 : i + 1 + Nf, :-1] for i in idx])
            utilde0toNfminus1.extend(
                [
                    torch.stack(
                        (u_ref[i : i + Nf, 0], x[i : i + Nf, 4] + u_ref[i : i + Nf, 1]),
                        dim=1,
                    )
                    for i in idx
                ]
            )

        self.xtilde0 = torch.stack(xtilde0)
        self.utilde0toNfminus1 = torch.stack(utilde0toNfminus1)
        self.xtilde1toNf = torch.stack(xtilde1toNf)
        self.Nf = Nf

        # translate to origin and rotate so that the initial heading is pi/2
        # position0 = self.xtilde0[:, :, :2]  # shape (N, 1, 2)
        # rot_angle = torch.pi / 2 - self.xtilde0[:, :, 2]
        # rot_matrix = torch.stack(
        #     (
        #         torch.stack((torch.cos(rot_angle), torch.sin(rot_angle)), dim=2),
        #         torch.stack((-torch.sin(rot_angle), torch.cos(rot_angle)), dim=2),
        #     ),
        #     dim=2,
        # )

        # self.xtilde0[:, :, :2] -= position0
        # self.xtilde0[:, :, 2] += rot_angle
        # self.xtilde0[:, :, :2] = torch.matmul(
        #     self.xtilde0[:, :, :2].unsqueeze(2), rot_matrix
        # ).squeeze(2)
        # self.xtilde1toNf[:, :, :2] -= position0
        # self.xtilde1toNf[:, :, 2] += rot_angle
        # self.xtilde1toNf[:, :, :2] = torch.matmul(
        #     self.xtilde1toNf[:, :, :2].unsqueeze(2), rot_matrix
        # ).squeeze(2)

    def __len__(self) -> int:
        return self.xtilde0.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.xtilde0[idx], self.utilde0toNfminus1[idx], self.xtilde1toNf[idx]


class SysidTrainDataset(SysidDataset):
    def __init__(self, file_paths: list[str]) -> None:
        super().__init__(file_paths, 1)


class SysidTestDataset(SysidDataset):
    def __init__(self, file_paths: list[str], Nf: int) -> None:
        super().__init__(file_paths, Nf)


class ControlDataset(Dataset):
    x0: torch.Tensor  # shape (N, 1, nx)
    xref0toNf: torch.Tensor  # shape (N, Nf+1, nx)
    uref0toNfminus1: torch.Tensor  # shape (N, Nf, nu)

    def __init__(self, file_paths: list[str]) -> None:
        super().__init__()
        x0, xref0toNf, uref0toNfminus1 = [], [], []
        Nf = None
        for path in file_paths:
            _, x, xref, uref = load_data(path, format="torch")
            if Nf is None:
                Nf = uref.shape[1] // 2
            else:
                assert Nf == uref.shape[1] // 2, f"Nf is not consistent in file {path}"

            idx = torch.arange(0, x.shape[0] - 1)
            x0.extend([x[i : i + 1] for i in idx])
            tpr = [xref[i].reshape(Nf + 1, -1) for i in idx]
            for i in idx:
                offset = x[i, 2] - torch.pi
                tpr[i][:, 2] = teds_projection(tpr[i][:, 2], offset)

            xref0toNf.extend(tpr)
            uref0toNfminus1.extend([uref[i].reshape(Nf, -1) for i in idx])

        self.x0 = torch.stack(x0)
        self.xref0toNf = torch.stack(xref0toNf)
        self.uref0toNfminus1 = torch.stack(uref0toNfminus1)

        # translate to origin and rotate so that the initial heading is pi/2
        # position0 = self.x0[:, :, :2]  # shape (N, 1, 2)
        # rot_angle = torch.pi / 2 - self.x0[:, :, 2]  # shape (N, 1)
        # rot_matrix = torch.stack(
        #     (
        #         torch.stack((torch.cos(rot_angle), torch.sin(rot_angle)), dim=2),
        #         torch.stack((-torch.sin(rot_angle), torch.cos(rot_angle)), dim=2),
        #     ),
        #     dim=3,
        # )  # shape (N,1,2,2)

        # self.x0[:, :, :2] -= position0
        # self.x0[:, :, 2] += rot_angle
        # self.x0[:, :, :2] = torch.matmul(
        #     self.x0[:, :, :2].unsqueeze(2), rot_matrix
        # ).squeeze(2)
        # self.xref0toNf[:, :, :2] -= position0
        # self.xref0toNf[:, :, 2] += rot_angle
        # self.xref0toNf[:, :, :2] = torch.matmul(
        #     self.xref0toNf[:, :, :2].unsqueeze(2), rot_matrix
        # ).squeeze(2)

    def __len__(self) -> int:
        return self.x0.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.x0[idx], self.xref0toNf[idx], self.uref0toNfminus1[idx]


def get_sysid_loaders(file_paths: list[str]):
    d = SysidTrainDataset(file_paths)
    train_data, val_data = random_split(
        d, [int(len(d) * 0.8), len(d) - int(len(d) * 0.8)]
    )
    train_loader = DataLoader(train_data, batch_size=len(train_data), shuffle=True)
    val_loader = DataLoader(val_data, batch_size=len(val_data), shuffle=True)
    return train_loader, val_loader


def get_control_loaders(file_paths: list[str], batch_sizes=(0, 0)):
    d = ControlDataset(file_paths)
    train_data, val_data = random_split(
        d, [int(len(d) * 0.8), len(d) - int(len(d) * 0.8)]
    )
    train_loader = DataLoader(
        train_data,
        batch_size=len(train_data) if batch_sizes[0] == 0 else batch_sizes[0],
        shuffle=True,
    )
    val_loader = DataLoader(
        val_data,
        batch_size=len(val_data) if batch_sizes[1] == 0 else batch_sizes[1],
        shuffle=True,
    )
    return train_loader, val_loader


if __name__ == "__main__":
    dataset = SysidDataset(["bruh.csv"], Nf=2)
    print(f"x0 shape: {dataset.x0.shape}")
    print(f"u0toNf shape: {dataset.u0toNf.shape}")
    print(f"x1toNf shape: {dataset.x1toNf.shape}")
    print(f"x0: {dataset.x0}")
    print(f"u0toNf: {dataset.u0toNf}")
    print(f"x1toNf: {dataset.x1toNf}")
