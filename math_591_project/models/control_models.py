# Copyright (c) 2023 Tudor Oancea
from abc import ABC, abstractmethod
from typing import Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from icecream import ic

__all__ = [
    "KIN4_NXTILDE",
    "KIN4_NUTILDE",
    "KIN4_NX",
    "DYN6_NXTILDE",
    "DYN6_NUTILDE",
    "DYN6_NX",
    "ODE",
    "Kin4ODE",
    "Dyn6ODE",
    "BlackboxKin4ODE",
    "BlackboxDyn6ODE",
    "RK4",
    "ControlDiscreteModel",
    "MLPControlPolicy",
    "OpenLoop",
    "ClosedLoop",
]


KIN4_NXTILDE = 4
KIN4_NUTILDE = 2
KIN4_NX = 5
DYN6_NXTILDE = 6
DYN6_NUTILDE = 2
DYN6_NX = 7


class ODE(nn.Module):
    nxtilde: int
    nutilde: int

    def __init__(self, nxtilde: int, nutilde: int):
        super().__init__()
        self.nxtilde = nxtilde
        self.nutilde = nutilde

    def forward(self, xtilde: torch.Tensor, utilde: torch.Tensor):
        """
        :param xtilde: shape (batch_size, nxtilde)
        :param utilde: shape (batch_size, nutilde)
        :return xtildedot: shape (batch_size, nxtilde)
        """
        assert (
            xtilde.shape[0] == utilde.shape[0]
        ), f"Batch size of x and u must match but are {xtilde.shape[0]} and {utilde.shape[0]}"
        assert (
            len(xtilde.shape) == 2 and xtilde.shape[1] == self.nxtilde
        ), f"x must have shape (batch_size, {self.nxtilde}) but has shape {xtilde.shape}"
        assert (
            len(utilde.shape) == 2 and utilde.shape[1] == self.nutilde
        ), f"u must have shape (batch_size, {self.nutilde}) but has shape {utilde.shape}"


class Kin4ODE(ODE):
    def __init__(
        self,
        m=223.0,
        l_R=0.8,
        l_F=0.4,
        C_m1=3500.0,
        C_m2=0.0,
        C_r0=0.0,
        C_r1=0.0,
        C_r2=3.5,
    ):
        super().__init__(nxtilde=KIN4_NXTILDE, nutilde=KIN4_NUTILDE)
        self.m = m
        self.l_R = l_R
        self.l_F = l_F
        self.C_m1 = nn.Parameter(torch.tensor(C_m1))
        self.C_m2 = nn.Parameter(torch.tensor(C_m2))
        self.C_r0 = nn.Parameter(torch.tensor(C_r0))
        self.C_r1 = nn.Parameter(torch.tensor(C_r1))
        self.C_r2 = nn.Parameter(torch.tensor(C_r2))

    def forward(self, xtilde: torch.Tensor, utilde: torch.Tensor) -> torch.Tensor:
        """
        :param xtilde: shape (batch_size, nxtilde) = (batch_size, 4)
        :param utilde: shape (batch_size, nutilde) = (batch_size, 2)
        :return xtildedot: shape (batch_size, nxtilde) = (batch_size, 4)
        """

        super().forward(xtilde, utilde)
        beta = torch.arctan(torch.tan(utilde[:, 1]) * self.l_R / (self.l_R + self.l_F))
        F_x = (
            (self.C_m1 - self.C_m2 * xtilde[:, 3]) * utilde[:, 0]
            - self.C_r0
            - self.C_r1 * xtilde[:, 3]
            - self.C_r2 * xtilde[:, 3] ** 2
        )
        return torch.stack(
            [
                xtilde[:, 3] * torch.cos(xtilde[:, 2] + beta),
                xtilde[:, 3] * torch.sin(xtilde[:, 2] + beta),
                xtilde[:, 3] * torch.sin(beta) / self.l_R,
                F_x / self.m,
            ],
            dim=1,
        )


class Dyn6ODE(ODE):
    def __init__(
        self,
        m=223.0,
        l_R=0.8,
        l_F=0.4,
        I_z=78.79,
        C_m1=3500.0,
        C_m2=0.0,
        C_r0=0.0,
        C_r1=0.0,
        C_r2=3.5,
        B_R=11.5,
        C_R=1.98,
        D_R=1.67,
        B_F=11.5,
        C_F=1.98,
        D_F=1.67,
    ):
        super().__init__(nxtilde=DYN6_NXTILDE, nutilde=DYN6_NUTILDE)
        self.m = m
        self.l_R = l_R
        self.l_F = l_F
        self.I_z = I_z
        self.C_m1 = nn.Parameter(torch.tensor(C_m1))
        self.C_m2 = nn.Parameter(torch.tensor(C_m2))
        self.C_r0 = nn.Parameter(torch.tensor(C_r0))
        self.C_r1 = nn.Parameter(torch.tensor(C_r1))
        self.C_r2 = nn.Parameter(torch.tensor(C_r2))
        self.B_R = nn.Parameter(torch.tensor(B_R))
        self.C_R = nn.Parameter(torch.tensor(C_R))
        self.D_R = nn.Parameter(torch.tensor(D_R))
        self.B_F = nn.Parameter(torch.tensor(B_F))
        self.C_F = nn.Parameter(torch.tensor(C_F))
        self.D_F = nn.Parameter(torch.tensor(D_F))

    def forward(self, xtilde: torch.Tensor, utilde: torch.Tensor) -> torch.Tensor:
        """
        :param xtilde: shape (batch_size, nxtilde) = (batch_size, 6)
        :param utilde: shape (batch_size, nutilde) = (batch_size, 2)
        :return xtildedot: shape (batch_size, nxtilde) = (batch_size, 6)
        """
        super().forward(xtilde, utilde)
        F_x = (
            (self.C_m1 - self.C_m2 * xtilde[:, 3]) * utilde[:, 0]
            - self.C_r0
            - self.C_r1 * xtilde[:, 3]
            - self.C_r2 * xtilde[:, 3] ** 2
        )
        F_y_R = self.D_R * torch.sin(
            self.C_R
            * torch.atan(
                self.B_R
                * torch.atan(
                    (xtilde[:, 4] - self.l_R * xtilde[:, 5]) / (1e-6 + xtilde[:, 3])
                )
            )
        )
        F_y_F = self.D_F * torch.sin(
            self.C_F
            * torch.atan(
                self.B_F
                * (
                    torch.atan(
                        (xtilde[:, 4] + self.l_F * xtilde[:, 5]) / (1e-6 + xtilde[:, 3])
                    )
                    - utilde[:, 1]
                )
            )
        )
        return torch.stack(
            [
                xtilde[:, 3] * torch.cos(xtilde[:, 2])
                - xtilde[:, 4] * torch.sin(xtilde[:, 2]),
                xtilde[:, 3] * torch.sin(xtilde[:, 2])
                + xtilde[:, 4] * torch.cos(xtilde[:, 2]),
                xtilde[:, 5],
                F_x / self.m
                + xtilde[:, 4] * xtilde[:, 5]
                - F_y_F * torch.sin(utilde[:, 1]) / self.m,
                (F_y_R + F_y_F * torch.cos(utilde[:, 1])) / self.m
                - xtilde[:, 3] * xtilde[:, 5],
                (F_y_F * self.l_F * torch.cos(utilde[:, 1]) - F_y_R * self.l_R)
                / (1e-6 + self.I_z),
            ],
            dim=1,
        )


class BlackboxKin4ODE(ODE):
    def __init__(self, net: nn.Module):
        super().__init__(nxtilde=KIN4_NXTILDE, nutilde=KIN4_NUTILDE)
        # check that net is compatible with the input size
        with torch.no_grad():
            try:
                output = net(torch.ones(2, self.nxtilde + self.nutilde))
                assert output.shape == (
                    2,
                    self.nxtilde,
                ), "net must output a tensor of shape (batch, nxtilde)=(batch, 4)"
            except:
                raise ValueError(
                    "net must take as input a tensor of shape (batch, nxtilde+nutilde)=(batch, 6)"
                )

        self.net = net

    def forward(self, xtilde: torch.Tensor, utilde: torch.Tensor) -> torch.Tensor:
        """
        :param xtilde: shape (batch_size, nxtilde) = (batch_size, 4)
        :param utilde: shape (batch_size, nutilde) = (batch_size, 2)
        :return xtildedot: shape (batch_size, nxtilde) = (batch_size, 4)
        """
        super().forward(xtilde, utilde)
        return self.net(torch.cat([xtilde, utilde], dim=1))


class BlackboxDyn6ODE(ODE):
    def __init__(self, net: nn.Module):
        super().__init__(nxtilde=DYN6_NXTILDE, nutilde=DYN6_NUTILDE)
        # check that net is compatible with the input size
        with torch.no_grad():
            batch_size = 2
            try:
                output = net(torch.ones(batch_size, self.nxtilde + self.nutilde - 3))
            except:
                raise ValueError(
                    "net must take as input a tensor of shape (batch, nx+nu-3)=(batch, 5)"
                )
            assert output.shape == (
                batch_size,
                self.nxtilde - 3,
            ), "net must output a tensor of shape (batch, nxtilde-3)=(batch, 3)"

        self.net = net

    def forward(self, xtilde: torch.Tensor, utilde: torch.Tensor) -> torch.Tensor:
        """
        :param xtilde: shape (batch_size, nxtilde) = (batch_size, 6)
        :param utilde: shape (batch_size, nutilde) = (batch_size, 2)
        :return xtildedot: shape (batch_size, nxtilde) = (batch_size, 6)"""
        super().forward(xtilde, utilde)
        phi = xtilde[:, 2:3]
        v_x = xtilde[:, 3:4]
        v_y = xtilde[:, 4:5]
        r = xtilde[:, 5:6]
        return torch.cat(
            (
                v_x * torch.cos(phi) - v_y * torch.sin(phi),
                v_x * torch.sin(phi) + v_y * torch.cos(phi),
                r,
                self.net(torch.cat([xtilde[:, 3:], utilde], dim=1)),
            ),
            dim=1,
        )


class BaseDiscretization(ABC, nn.Module):
    nxtilde: int
    nutilde: int
    ode: nn.Module
    dt: float

    def __init__(self, nxtilde: int, nutilde: int, ode: nn.Module, dt: float) -> None:
        super().__init__()
        self.nxtilde = nxtilde
        self.nutilde = nutilde
        self.ode = ode
        self.dt = dt

    @abstractmethod
    def forward(self, xtilde: torch.Tensor, utilde: torch.Tensor):
        """
        :param xtilde: shape (batch_size, nxtilde)
        :param utilde: shape (batch_size, nutilde)
        :return xtildenext: shape (batch_size, nxtilde)
        """
        assert (
            xtilde.shape[0] == utilde.shape[0]
        ), f"batch_size of x and u must be equal but are respectively {xtilde.shape[0]} and {utilde.shape[0]}"
        assert len(xtilde.shape) == 2 and xtilde.shape[1:] == (
            self.nxtilde,
        ), f"x.shape must be (batch_size, {self.nxtilde}) but is {xtilde.shape}"
        assert len(utilde.shape) == 2 and utilde.shape[1:] == (
            self.nutilde,
        ), f"u.shape must be (batch_size, {self.nutilde}) but is {utilde.shape}"


class RK4(BaseDiscretization):
    def __init__(self, nxtilde: int, nutilde: int, ode: nn.Module, dt: float):
        super().__init__(nxtilde, nutilde, ode, dt)

    def forward(self, xtilde: torch.Tensor, utilde: torch.Tensor) -> torch.Tensor:
        """
        :param xtilde: shape (batch_size, nxtilde)
        :param utilde: shape (batch_size, nutilde)
        :return xtildenext: shape (batch_size, nxtilde)
        """
        super().forward(xtilde, utilde)
        k1 = self.ode(xtilde, utilde)
        k2 = self.ode(xtilde + 0.5 * self.dt * k1, utilde)
        k3 = self.ode(xtilde + 0.5 * self.dt * k2, utilde)
        k4 = self.ode(xtilde + self.dt * k3, utilde)
        output = xtilde + self.dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        return output


class ControlDiscreteModel(nn.Module):
    def __init__(self, model: BaseDiscretization, nx: int, nu: int):
        super().__init__()
        self.model = model
        self.nx = nx
        self.nu = nu

    def forward(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """
        x_k = (xtilde_k, delta_{k-1}), u_k = (T_k, ddelta_k)
        :param x: shape (batch_size, nx)
        :param u: shape (batch_size, nu)
        :return xnext: shape (batch_size, nx)
        """
        assert x.shape[1] == self.nx
        assert u.shape[1] == self.nu
        delta = x[:, -1:] + u[:, 1:2]
        return torch.cat(
            (
                self.model(x[:, :-1], torch.cat((u[:, 0:1], delta), dim=1)),
                delta,
            ),
            dim=1,
        )


class ControlPolicy(ABC, nn.Module):
    nx: int
    nu: int
    Nf: int

    def __init__(self, nx: int, nu: int, Nf: int) -> None:
        super().__init__()
        self.nx = nx
        self.nu = nu
        self.Nf = Nf

    @abstractmethod
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


class OpenLoop(nn.Module):
    model: Union[BaseDiscretization, ControlDiscreteModel]
    Nf: int

    def __init__(
        self, model: Union[BaseDiscretization, ControlDiscreteModel], Nf: int
    ) -> None:
        super().__init__()
        self.model = model
        self.Nf = Nf

    def forward(self, x0: torch.Tensor, u0toNfminus1: torch.Tensor):
        """
        :param x0: (batch_size, 1, nx)
        :param u0toNfminus1: (batch_size, Nf, nu)
        :return x1toNf: (batch_size, Nf, nx)
        """
        assert u0toNfminus1.shape[1] == self.Nf
        assert x0.shape[0] == u0toNfminus1.shape[0]
        x1toNf = [self.model(x0[:, 0, :], u0toNfminus1[:, 0, :])]
        for i in range(1, self.Nf):
            x1toNf.append(self.model(x1toNf[-1], u0toNfminus1[:, i, :]))

        return torch.stack(x1toNf, dim=1)


class ClosedLoop(nn.Module):
    system_model: ControlDiscreteModel
    control_model: ControlPolicy

    def __init__(
        self, system_model: ControlDiscreteModel, control_model: ControlPolicy
    ) -> None:
        super().__init__()
        self.system_model = system_model
        self.control_model = control_model

    def forward(self, x0: torch.Tensor, xref0toNf: torch.Tensor):
        Nf = xref0toNf.shape[1] - 1

        x = [x0.squeeze()]
        u = []
