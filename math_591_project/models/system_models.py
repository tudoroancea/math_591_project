# Copyright (c) 2023 Tudor Oancea
import torch
import torch.nn as nn
from icecream import ic

__all__ = [
    "Model",
    "ContinuousModel",
    "DiscreteModel",
    "Kin4",
    "Dyn6",
    "NeuralMixin",
    "NeuralDyn6",
    "RK4",
    "OpenLoop",
    "ode_from_string",
]


class Model(nn.Module):
    state_dim: int
    control_dim: int

    # def __init__(self, state_dim: int, control_dim: int):
    #     super().__init__()
    #     self.state_dim = state_dim
    #     self.control_dim = control_dim

    def forward(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """
        :param x: shape (batch_size, state_dim)
        :param u: shape (batch_size, control_dim)
        :return xnext or xdot: shape (batch_size, state_dim)
        """
        assert len(x.shape) == 2, f"x must be 2D but is {len(x.shape)}D"
        assert len(u.shape) == 2, f"u must be 2D but is {len(u.shape)}D"
        assert (
            x.shape[0] == u.shape[0]
        ), f"batch sizes must match but are {x.shape[0]} for x and {u.shape[0]} for u"
        assert (
            x.shape[1] == self.state_dim
        ), f"x.shape[1] must be {self.state_dim} but is {x.shape[1]}"
        assert (
            u.shape[1] == self.control_dim
        ), f"u.shape[1] must be {self.control_dim} but is {u.shape[1]}"


class ContinuousModel(Model):
    # def __init__(self, state_dim: int, control_dim: int):
    #     super().__init__(state_dim, control_dim)
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()


class DiscreteModel(Model):
    dt: float

    # def __init__(self, state_dim: int, control_dim: int, dt: float):
    #     super().__init__(state_dim, control_dim)
    #     self.dt = dt
    def __init__(self, dt: float, *args, **kwargs):
        super().__init__()
        self.dt = dt


class Kin4(ContinuousModel):
    state_dim = 4
    control_dim = 2

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
        super().__init__()
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


class Dyn6(ContinuousModel):
    state_dim = 6
    control_dim = 2
    B = 11.5
    C = 1.98
    D = 1670.0
    # B = 11.5
    # C = 1.4
    # D = 70.0

    def __init__(
        self,
        m=223.0,
        l_R=0.8,
        l_F=0.4,
        I_z=78.79,
        # C_m1=3300.0,
        # C_m2 = 40.0,
        # C_r0 = 100.0,
        # C_r1 = 1.0e4,
        # C_r2 = 3.0,
        C_m1=3500.0,
        C_m2=10.0,
        C_r0=0.1,
        C_r1=5.0,
        C_r2=2.2,
        # C_m1=8000.0,
        # C_m2=172.0,
        # C_r0=180.0,
        # C_r1=1e5,
        # C_r2=0.7,
        B_R=B,
        C_R=C,
        D_R=D,
        B_F=B,
        C_F=C,
        D_F=D,
    ):
        super().__init__()
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
            - self.C_r0 * torch.tanh(1000 * self.C_r0 * xtilde[:, 3])
            - self.C_r1 * xtilde[:, 3]
            - self.C_r2 * xtilde[:, 3] ** 2
        )
        alpha_R = torch.atan2(
            xtilde[:, 4] - self.l_R * xtilde[:, 5], xtilde[:, 3] + 1e-6
        )
        alpha_F = (
            torch.atan2(xtilde[:, 4] + self.l_F * xtilde[:, 5], xtilde[:, 3] + 1e-6)
            - utilde[:, 1]
        )
        # alpha_R = torch.atan(
        #     (xtilde[:, 4] - self.l_R * xtilde[:, 5]) / (xtilde[:, 3] + 1e-6)
        # )
        # alpha_F = (
        #     torch.atan((xtilde[:, 4] + self.l_F * xtilde[:, 5]) / (xtilde[:, 3] + 1e-6))
        #     - utilde[:, 1]
        # )
        alpha_R, alpha_F = -alpha_R, -alpha_F
        F_y_R = self.D_R * torch.sin(self.C_R * torch.atan(self.B_R * alpha_R))
        F_y_F = self.D_F * torch.sin(self.C_F * torch.atan(self.B_F * alpha_F))
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


class NeuralMixin:
    nin: int
    nout: int
    net: nn.Module

    def __init__(self, net: nn.Module) -> None:
        with torch.no_grad():
            batch_size = 2
            try:
                output = net(torch.zeros(batch_size, self.nin))
            except:
                raise ValueError(
                    f"net must take as input a tensor of shape (batch, {self.nin})"
                )
            assert output.shape == (
                batch_size,
                self.nout,
            ), f"net must output a tensor of shape (batch, {self.nout})"

        self.net = net


class NeuralDyn6(ContinuousModel, NeuralMixin):
    state_dim = 6
    control_dim = 2
    nin = state_dim + control_dim - 3
    nout = state_dim - 3

    def __init__(self, net: nn.Module):
        # check that net is compatible with the input size
        # with torch.no_grad():
        #     batch_size = 2
        #     try:
        #         output = net(torch.ones(batch_size, self.nxtilde + self.nutilde - 3))
        #     except:
        #         raise ValueError(
        #             "net must take as input a tensor of shape (batch, nx+nu-3)=(batch, 5)"
        #         )
        #     assert output.shape == (
        #         batch_size,
        #         self.nxtilde - 3,
        #     ), "net must output a tensor of shape (batch, nxtilde-3)=(batch, 3)"

        # self.net = net
        ContinuousModel.__init__(self)
        NeuralMixin.__init__(self, net)

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


class RK4(DiscreteModel):
    def __init__(self, ode: ContinuousModel, dt: float):
        super().__init__(dt)
        self.state_dim = ode.state_dim
        self.control_dim = ode.control_dim
        self.ode = ode

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
    model: Model
    Nf: int

    def __init__(self, model: Model, Nf: int) -> None:
        super().__init__()
        self.model = model
        self.Nf = Nf

    def forward(self, x0: torch.Tensor, u0toNfminus1: torch.Tensor):
        """
        :param x0: (batch_size, 1, nx)
        :param u0toNfminus1: (batch_size, Nf, nu)
        :return x1toNf: (batch_size, Nf, nx)
        """
        assert (
            x0.shape[0] == u0toNfminus1.shape[0]
        ), f"batch size must be the same but are {x0.shape[0]} for x0 (total shape {x0.shape})and {u0toNfminus1.shape[0]} for u0toNfminus1 (total shape {u0toNfminus1.shape})"
        assert u0toNfminus1.shape[1] == self.Nf
        x1toNf = [self.model(x0[:, 0, :], u0toNfminus1[:, 0, :])]
        for i in range(1, self.Nf):
            x1toNf.append(self.model(x1toNf[-1], u0toNfminus1[:, i, :]))

        return torch.stack(x1toNf, dim=1)


ode_from_string = {
    "kin4": Kin4,
    "Kin4": Kin4,
    "Dyn6": Dyn6,
    "dyn6": Dyn6,
    "NeuralDyn6": NeuralDyn6,
    "neural_dyn6": NeuralDyn6,
}
