# Copyright (c) 2023 Tudor Oancea
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from icecream import ic

from .system_models import *

__all__ = [
    "KIN4_NX",
    "DYN6_NX",
    "LiftedDiscreteModel",
    "MLPControlPolicy",
    "OpenLoop",
]

KIN4_NX = 5
DYN6_NX = 7


class LiftedDiscreteModel(DiscreteModel):
    def __init__(self, model: DiscreteModel, nx: int, nu: int):
        super().__init__(model.dt)
        self.state_dim = model.state_dim + 1
        self.control_dim = model.control_dim
        self.model = model

    def forward(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """
        x_k = (xtilde_k, delta_{k-1}), u_k = (T_k, ddelta_k)
        :param x: shape (batch_size, nx)
        :param u: shape (batch_size, nu)
        :return xnext: shape (batch_size, nx)
        """
        delta = x[:, -1:] + u[:, 1:2]
        return torch.cat(
            (
                self.model(x[:, :-1], torch.cat((u[:, 0:1], delta), dim=1)),
                delta,
            ),
            dim=1,
        )


class ControlPolicy(nn.Module):
    nx: int
    nu: int
    Nf: int

    def __init__(self, nx: int, nu: int, Nf: int) -> None:
        super().__init__()
        self.nx = nx
        self.nu = nu
        self.Nf = Nf

    def forward(self, x0: torch.Tensor, xref0toNf: torch.Tensor) -> torch.Tensor:
        """
        :param x0: shape (batch_size, nx)
        :param xref0toNf: shape (batch_size, Nf+1, nx)
        """
        assert (
            x0.shape[0] == xref0toNf.shape[0]
        ), f"batch_size of x0 and xref0toNf must be equal but are respectively {x0.shape[0]} and {xref0toNf.shape[0]}"
        assert (
            len(x0.shape) == 3 and x0.shape[2] == self.nx and x0.shape[1] == 1
        ), f"x0.shape must be (batch_size, 1, {self.nx}) but is {x0.shape}"
        assert len(xref0toNf.shape) == 3 and xref0toNf.shape[1:] == (
            self.Nf + 1,
            4,
        ), f"xref0toNf.shape must be (batch_size, {self.Nf + 1}, {4}) but is {xref0toNf.shape}"


class MLPControlPolicy(ControlPolicy):
    def __init__(self, nx: int, nu: int, Nf: int, mlp: nn.Module):
        super().__init__(nx, nu, Nf)
        self.mlp = mlp
        try:
            output = self.mlp(torch.zeros(2, nx - 3 + (Nf + 1) * 4))
        except:
            raise ValueError(
                f"mlp must take as input a tensor of shape (batch_size, nx + (Nf+1)*4)=(batch_size,{nx+(Nf+1)*4})"
            )
        assert output.shape[1] == nu * Nf, (
            f"mlp must output a tensor of shape (batch_size, nu*Nf)=(batch_size,{nu*Nf})"
            f" but is {output.shape}"
        )
        self.output_scaling = torch.tensor(
            [[[1.0, np.deg2rad(68.0) / 20.0]]], dtype=torch.float32
        )

    def forward(self, x0: torch.Tensor, xref0toNf: torch.Tensor) -> torch.Tensor:
        """
        :param x0: shape (batch_size, 1, nx)
        :param xref0toNf: shape (batch_size, Nf+1, nx)
        :return u0toNfminus1: shape (batch_size, Nf, nu)
        """
        super().forward(x0, xref0toNf)
        batch_size = x0.shape[0]

        # move output_scaling to the same device as x0 (only done once)
        # (we can't do it in __init__ because we don't know the device of x0 at this time)
        if self.output_scaling.device != x0.device:
            self.output_scaling = self.output_scaling.to(x0.device)

        # compute input of mlp as the transformation of the reference poses into the local frame of the first pose
        philoc = xref0toNf[:, :, 2] - x0[:, :, 2]  # shape (batch_size, Nf+1)
        rot_matrix = torch.stack(
            (
                torch.stack(
                    (torch.cos(philoc), -torch.sin(philoc)), dim=2
                ),  # shape (batch_size, Nf+1, 2)
                torch.stack(
                    (torch.sin(philoc), torch.cos(philoc)), dim=2
                ),  # shape (batch_size, Nf+1, 2)
            ),
            dim=3,
        )  # shape (batch_size, Nf+1, 2, 2)
        XYloc = torch.matmul(
            torch.unsqueeze(xref0toNf[:, :, :2] - x0[:, :, :2], dim=2),
            rot_matrix,
        )  # shape (batch_size, Nf+1, 1, 2)
        input = torch.cat(
            (
                x0[:, 0, 3:],  # shape (batch_size, 2)
                torch.cat(
                    (XYloc.squeeze(2), philoc.unsqueeze(2), xref0toNf[:, :, 3:]), dim=2
                ).view(batch_size, -1),
            ),
            dim=1,
        )

        # apply mlp and scale and reshape output
        return (
            F.tanh(self.mlp(input)).view(batch_size, self.Nf, self.nu)
            * self.output_scaling
        )
