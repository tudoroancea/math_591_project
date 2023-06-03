# Copyright (c) 2023 Tudor Oancea
import os
import sys

import numpy as np
import torch
from models import *


def main():
    Nbatch = 1000
    base_model = sys.argv[1]
    nx = KIN4_NXTILDE if base_model == "kin4" else DYN6_NXTILDE
    nu = KIN4_NU if base_model == "kin4" else DYN6_NU
    Nf = 40
    dt = 1 / 20
    dumpdir = "data/sysid/" + base_model
    m = 223.0
    l_R = 0.8
    l_F = 0.4
    I_z = 78.79
    C_m = 3500.0
    C_r0 = 0.0
    C_r1 = 0.0
    C_r2 = 3.5
    B_R = 20.0
    C_R = 0.8
    D_R = 4.0
    B_F = 17.0
    C_F = 2.0
    D_F = 4.2

    # model =====================================================================
    open_loop_model = OpenLoop(
        model=RK4(
            nx=nx,
            nu=nu,
            ode=Kin4ODE(m=m, l_R=l_R, l_F=l_F, C_m=C_m, C_r0=C_r0, C_r1=C_r1, C_r2=C_r2)
            if base_model == "kin4"
            else Dyn6ODE(
                m=m,
                l_R=l_R,
                l_F=l_F,
                C_m=C_m,
                C_r0=C_r0,
                C_r1=C_r1,
                C_r2=C_r2,
                I_z=I_z,
                B_R=B_R,
                C_R=C_R,
                D_R=D_R,
                B_F=B_F,
                C_F=C_F,
                D_F=D_F,
            ),
            dt=dt,
        ),
        Nf=Nf,
    )

    # create a batch of random initial states and controls to apply ============
    x = torch.zeros(Nbatch, Nf + 1, nx)
    x[:, 0, 2] = torch.pi / 2
    x[:, 0, 3:] = torch.cat(
        [
            torch.normal(5, 5 / 3, size=(Nbatch, 1)),  # v_x
        ]
        + (
            [
                torch.normal(0, 0.8 / 3, size=(Nbatch, 1)),  # v_y
                torch.normal(0, 1 / 3, size=(Nbatch, 1)),  # r
            ]
            if base_model == "dyn6"
            else []
        ),
        dim=1,
    )

    ddelta = torch.rand(Nbatch, Nf - 1) * 2 * np.deg2rad(80) - np.deg2rad(80)
    u = torch.stack(
        [
            torch.rand(Nbatch, Nf) * 2 - 1,
            torch.cat(
                [
                    torch.zeros(Nbatch, 1),
                    torch.clamp(
                        torch.cumsum(dt * ddelta, dim=1),
                        -np.deg2rad(40.0),
                        np.deg2rad(40.0),
                    ),
                ],
                dim=1,
            ),
        ],
        dim=2,
    )

    # run open loop model ======================================================
    x[:, 1:] = open_loop_model(x[:, 0], u)
    timestamps = torch.arange(Nf + 1) * dt
    u = torch.cat([u, torch.zeros(Nbatch, 1, nu)], dim=1)

    # dump every sequence of states and controls to a CSV file ==================
    if not os.path.exists(dumpdir):
        os.makedirs(dumpdir)
    else:
        print("Removing old CSV files...")
        # use glob to remove all CSV files in the directory
        for f in os.listdir(dumpdir):
            if f.endswith(".csv"):
                os.remove(os.path.join(dumpdir, f))

    for i in range(Nbatch):
        with open(f"{dumpdir}/episode{i}.csv", "w") as f:
            f.write(
                "timestamp,X,Y,phi,v_x"
                + (",v_y,r" if base_model == "dyn6" else "")
                + ",T,delta\n"
            )
            for j in range(Nf + 1):
                f.write(
                    f"{timestamps[j]},"
                    + ",".join([str(x[i, j, k].item()) for k in range(nx)])
                    + ","
                    + ",".join([str(u[i, j, k].item()) for k in range(nu)])
                    + "\n"
                )


if __name__ == "__main__":
    main()
