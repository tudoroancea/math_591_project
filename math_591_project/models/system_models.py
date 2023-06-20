# Copyright (c) 2023 Tudor Oancea
from abc import ABC, abstractmethod
from typing import Union

import torch
import torch.nn as nn
from icecream import ic

__all__ = [
    "KIN4_NXTILDE",
    "KIN4_NUTILDE",
    "DYN6_NXTILDE",
    "DYN6_NUTILDE",
    "ODE",
    "Kin4ODE",
    "Dyn6ODE",
    "GrayboxKin4ODE",
    "GrayboxDyn6ODE",
    "RK4",
    "ControlDiscreteModel",
    "MLPControlPolicy",
    "OpenLoop",
    "ClosedLoop",
]


KIN4_NXTILDE = 4
KIN4_NUTILDE = 2
DYN6_NXTILDE = 6
DYN6_NUTILDE = 2

class Model(nn.Module):
    state_dim:int
    control_dim:int
    def __init__(self, state_dim:int, control_dim:int):
        super().__init__()
        self.state_dim = state_dim
        self.control_dim = control_dim

class ContinuousModel(Model):
    def __init__(self, state_dim:int, control_dim:int):
        super().__init__(state_dim, control_dim)

class DiscreteModel(Model):
    dt: float
    def __init__(self, state_dim:int, control_dim:int, dt:float):
        super().__init__(state_dim, control_dim)
        self.dt = dt

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
            - self.C_r0 * torch.tanh(self.C_r1 * xtilde[:, 3])
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
            - self.C_r0 * torch.tanh(self.C_r1 * xtilde[:, 3])
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


class GrayboxKin4ODE(ODE):
    def __init__(self, net: nn.Module):
        super().__init__(nxtilde=KIN4_NXTILDE, nutilde=KIN4_NUTILDE)
        # check that net is compatible with the input size
        with torch.no_grad():
            try:
                output = net(torch.ones(2, self.nxtilde + self.nutilde))
            except:
                raise ValueError(
                    "net must take as input a tensor of shape (batch, nxtilde+nutilde)=(batch, 6)"
                )
            assert output.shape == (
                2,
                self.nxtilde,
            ), "net must output a tensor of shape (batch, nxtilde)=(batch, 4)"

        self.net = net

    def forward(self, xtilde: torch.Tensor, utilde: torch.Tensor) -> torch.Tensor:
        """
        :param xtilde: shape (batch_size, nxtilde) = (batch_size, 4)
        :param utilde: shape (batch_size, nutilde) = (batch_size, 2)
        :return xtildedot: shape (batch_size, nxtilde) = (batch_size, 4)
        """
        super().forward(xtilde, utilde)
        return self.net(torch.cat([xtilde, utilde], dim=1))


class GrayboxDyn6ODE(ODE):
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


class DiscreteModel(ABC, nn.Module())
class BaseDiscretization(ABC, nn.Module):
    nxtilde: int
    nutilde: int
    dt: float

    def __init__(self, nxtilde: int, nutilde: int, dt: float) -> None:
        super().__init__()
        self.nxtilde = nxtilde
        self.nutilde = nutilde
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
        super().__init__(nxtilde, nutilde, dt)
        self.ode

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
