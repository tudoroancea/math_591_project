from typing import Union

import numpy as np
import pandas as pd
import torch
from icecream import ic
from torch.utils.data import DataLoader, Dataset, random_split


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
    assert x.ndim == 1, "x must be 1D"
    if isinstance(x, np.ndarray):
        diffs = np.diff(x)
        diffs[diffs > 1.5 * np.pi] -= 2 * np.pi
        diffs[diffs < -1.5 * np.pi] += 2 * np.pi
        return np.insert(x[0] + np.cumsum(diffs), 0, x[0])
    else:
        diffs = torch.diff(x)
        diffs[diffs > 1.5 * np.pi] -= 2 * np.pi
        diffs[diffs < -1.5 * np.pi] += 2 * np.pi
        return torch.cat(
            (torch.atleast_1d(x[0]), x[0] + torch.cumsum(diffs, dim=0)), dim=0
        )


def load_data(file_path: str, format="numpy") -> torch.Tensor:
    # import data from csv file
    df = pd.read_csv(file_path)

    # read timestamps
    timestamps = df["timestamp"].to_numpy()

    # read state x=(X, Y, phi, v_x, v_y, r, last_delta)
    x_cols = (
        ["X", "Y", "phi", "v_x"]
        + (["v_y", "r"] if "v_y" in df.columns and "r" in df.columns else [])
        + ["last_delta"]
    )
    x = df[x_cols].to_numpy()

    # make phi values continuous
    # x[:, 2] = unwrapToPi(x[:, 2])
    x[:, 2] = wrapToPi(x[:, 2])

    # read reference trajectory xref=(X_ref, Y_ref, phi_ref, v_x_ref) and control inputs u_ref=(T, ddelta)
    Nf = int(df.columns[-1].split("_")[-1]) + 1
    xref_cols = []
    uref_cols = []
    for i in range(Nf):
        xref_cols.extend([f"X_ref_{i}", f"Y_ref_{i}", f"phi_ref_{i}", f"v_x_ref_{i}"])
        uref_cols.extend([f"T_{i}", f"ddelta_{i}"])
    xref_cols.extend([f"X_ref_{Nf}", f"Y_ref_{Nf}", f"phi_ref_{Nf}", f"v_x_ref_{Nf}"])
    xref = df[xref_cols].to_numpy()
    xref[:, 2] = wrapToPi(xref[:, 2])
    uref = df[uref_cols].to_numpy()

    # convert to torch tensors if desired
    if format == "torch":
        timestamps = torch.from_numpy(timestamps).to(dtype=torch.float32)
        x = torch.from_numpy(x).to(dtype=torch.float32)
        xref = torch.from_numpy(xref).to(dtype=torch.float32)
        uref = torch.from_numpy(uref).to(dtype=torch.float32)

    # find first index where v_x > 0.01 and only keep data after that
    idx = np.where(x[:, 3] > 0.01)[0][0]
    timestamps = timestamps[idx:]
    x = x[idx:]
    xref = xref[idx:]
    uref = uref[idx:]

    return timestamps, x, xref, uref


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
            _, x, _, uref = load_data(path, format="torch")
            idx = torch.arange(0, x.shape[0] - Nf)
            tpr1 = [x[i : i + 1, :-1] for i in idx]
            tpr2 = [x[i + 1 : i + 1 + Nf, :-1] for i in idx]
            for i in idx:
                phi = x[i : i + 1 + Nf, 2]
                phi = unwrapToPi(phi)
                x[i : i + 1 + Nf, 2] = phi
                tpr1[i][0, 2] = phi[0]
                tpr2[i][:, 2] = phi[1:]
                # x[i:i + 1 + Nf, 2] = wrapToPi(x[i:i + 1 + Nf, 2])
                # x[i : i + 1 + Nf, 2] = unwrapToPi(x[i : i + 1 + Nf, 2])
                # xtilde0.append(x[i : i + 1, :-1])
                # xtilde1toNf.append(x[i + 1 : i + 1 + Nf, :-1])

            xtilde0.extend(tpr1)
            xtilde1toNf.extend(tpr2)
            utilde0toNfminus1.extend(
                [
                    torch.stack(
                        (uref[i : i + Nf, 0], x[i : i + Nf, 4] + uref[i : i + Nf, 1]),
                        dim=1,
                    )
                    for i in idx
                ]
            )

        self.xtilde0 = torch.stack(xtilde0)
        self.utilde0toNfminus1 = torch.stack(utilde0toNfminus1)
        self.xtilde1toNf = torch.stack(xtilde1toNf)
        self.Nf = Nf

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
                # project phi_ref to (phi-pi, phi+pi]
                tpr[i][:, 2] = teds_projection(tpr[i][:, 2], x[i, 2] - torch.pi)
                tpr[i][:, 2] = unwrapToPi(tpr[i][:, 2])

            xref0toNf.extend(tpr)
            uref0toNfminus1.extend([uref[i].reshape(Nf, -1) for i in idx])

        self.x0 = torch.stack(x0)
        self.xref0toNf = torch.stack(xref0toNf)
        self.uref0toNfminus1 = torch.stack(uref0toNfminus1)

    def __len__(self) -> int:
        return self.x0.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.x0[idx], self.xref0toNf[idx], self.uref0toNfminus1[idx]


def get_sysid_loaders(file_paths: list[str], batch_sizes=(0, 0), Nf=1):
    d = SysidDataset(file_paths, Nf)
    train_data, val_data = random_split(
        d, [int(len(d) * 0.8), len(d) - int(len(d) * 0.8)]
    )
    train_loader = DataLoader(
        train_data,
        batch_size=len(train_data) if batch_sizes[0] == 0 else batch_sizes[0],
        shuffle=True,
        num_workers=8,
    )
    val_loader = DataLoader(
        val_data,
        batch_size=len(val_data) if batch_sizes[1] == 0 else batch_sizes[1],
        shuffle=True,
        num_workers=8,
    )
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
