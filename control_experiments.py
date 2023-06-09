# Copyright (c) 2023 Tudor Oancea
import json
import os

import lightning as L
import scienceplots
import torch
import torch.nn.functional as F
from data_visualization import *
from lightning import Fabric
from matplotlib import pyplot as plt
from track_database import *

from math_591_project.models import *
from math_591_project.utils import *

L.seed_everything(127)
try:
    plt.style.use(["science"])
except OSError:
    print("science style not found, using default style")
plt.rcParams.update({"font.size": 20})


def load_config(config_path):
    return json.load(open(config_path, "r"))


def load_control_test_data(config: dict):
    data_dir = config["testing"]["data_dir"]
    file_paths = [
        os.path.abspath(os.path.join(data_dir, file_path))
        for file_path in os.listdir(data_dir)
        if file_path.endswith(".csv")
    ]
    test_dataset = ControlDataset(file_paths)
    return test_dataset


@torch.no_grad()
def control_open_loop():
    config_path = "config/control/neural_dyn6.json"
    config = load_config(config_path)
    # load control dataset
    fabric = Fabric()
    test_dataset = load_control_test_data(config)
    # load system model
    dt = 1 / 20
    Nf = 40
    kin4_model = OpenLoop(
        model=LiftedDiscreteModel(
            model=RK4(
                ode=Kin4(),
                dt=dt,
            ),
        ),
        Nf=Nf,
    )
    blackbox_dyn6_model = OpenLoop(
        model=LiftedDiscreteModel(
            model=RK4(
                ode=NeuralDyn6(
                    net=MLP(
                        nin=NeuralDyn6.nin,
                        nout=NeuralDyn6.nout,
                        nhidden=config["system_model"]["nhidden"],
                        nonlinearity=config["system_model"]["nonlinearity"],
                    ),
                ),
                dt=dt,
            ),
        ),
        Nf=Nf,
    )
    control_model = MLPControlPolicy(
        mlp=MLP(
            nin=MLPControlPolicy.nin,
            nout=MLPControlPolicy.nout,
            nhidden=config["control_model"]["nhidden"],
            nonlinearity=config["control_model"]["nonlinearity"],
        ),
    )
    try:
        blackbox_dyn6_model.model.model.ode.net.load_state_dict(
            torch.load(config["system_model"]["input_checkpoint"], map_location="cpu")
        )
    except FileNotFoundError:
        print("No checkpoint found for system model, using random initialization")
    except RuntimeError:
        print("Checkpoint found for system model, but not compatible with current model")

    try:
        control_model.mlp.load_state_dict(
            torch.load(config["control_model"]["input_checkpoint"], map_location="cpu")
        )
    except FileNotFoundError:
        print("No checkpoint found for control model, using random initialization")
    except RuntimeError:
        print("Checkpoint found for control model, but not compatible with current model")
    kin4_model = fabric.setup(kin4_model)
    blackbox_dyn6_model = fabric.setup(blackbox_dyn6_model)
    blackbox_dyn6_model.eval()
    kin4_model.eval()
    control_model = fabric.setup(control_model)
    control_model.eval()
    # take a sample from the test dataset and apply Kin4 to the reference control

    id = 878
    x0, xref0toNf, uref0toNfminus1 = test_dataset[id]
    x0, xref0toNf, uref0toNfminus1 = (
        x0.unsqueeze(0).to(fabric.device),
        xref0toNf.unsqueeze(0).to(fabric.device),
        uref0toNfminus1.unsqueeze(0).to(fabric.device),
    )

    # to plot: ref with blackbox_dyn6_model, ref with kin4_model, dpc with blackbox_dyn6_model, dpc with kin4_model
    plot_u0toNf = []
    plot_x1toNf = []
    #  1. ref with blackbox_dyn6_model
    x1toNf = blackbox_dyn6_model(x0, uref0toNfminus1)
    plot_x1toNf.append(x1toNf.cpu().numpy())
    plot_u0toNf.append(uref0toNfminus1.cpu().numpy())
    #  2. ref with kin4_model
    x1toNf = kin4_model(x0[..., [0, 1, 2, 3, 6]], uref0toNfminus1)
    x1toNf = torch.cat(
        (
            x1toNf[..., [0, 1, 2, 3]],
            torch.full((1, Nf, 2), torch.nan, device=x1toNf.device),
            x1toNf[..., [4]],
        ),
        dim=-1,
    )
    plot_x1toNf.append(x1toNf.cpu().numpy())
    plot_u0toNf.append(uref0toNfminus1.cpu().numpy())
    #  3. dpc with blackbox_dyn6_model
    u0toNfminus1 = control_model(x0, xref0toNf)
    x1toNf = blackbox_dyn6_model(x0, u0toNfminus1)
    plot_x1toNf.append(x1toNf.cpu().numpy())
    plot_u0toNf.append(u0toNfminus1.cpu().numpy())

    # plot
    plot_u0toNf = np.concatenate(plot_u0toNf, axis=0)
    plot_x1toNf = np.concatenate(plot_x1toNf, axis=0)
    x0 = x0.squeeze(0).cpu().numpy()
    xref0toNf = xref0toNf.squeeze(0).cpu().numpy()
    plot_control_trajs(
        x0,
        xref0toNf,
        plot_u0toNf,
        plot_x1toNf,
        [
            "MPC+NeuralDyn6",
            "MPC+Kin4",
            "DPC+NeuralDyn6",
        ],
    )
    plt.savefig("experiments/control/dpc_ol.png", dpi=300)
    # plt.show()


def control_closed_loop():
    df = pd.read_csv("experiments/control/dpc_closed_loop.csv")
    method = df["method"].values
    X = df["X"].values
    Y = df["Y"].values
    T = df["T_0"].values
    delta = np.rad2deg(df["last_delta"].values + df["ddelta_0"].values)
    ddelta = np.rad2deg(df["ddelta_0"].values)
    # find the first index where the method changes
    id = np.argwhere(method[:-1] != method[1:]).ravel() + 1
    id = id[0]

    # load track
    track = load_track("fsds_competition_1")

    # create a 3x2 gridspec and use the left half for the traj, the right half for the control (T, delta, ddelta)
    fig = plt.figure(figsize=(20, 10))
    gs = fig.add_gridspec(3, 2)
    ax_traj = fig.add_subplot(gs[:, 0])
    ax_T = fig.add_subplot(gs[0, 1])
    ax_delta = fig.add_subplot(gs[1, 1])
    ax_ddelta = fig.add_subplot(gs[2, 1])

    # plot the trajectory
    ax_traj.plot(
        np.append(track.center_line[-1, 0], track.center_line[:, 0]),
        np.append(track.center_line[-1, 1], track.center_line[:, 1]),
        label="center line",
        color="black",
    )
    ax_traj.plot(X[:id], Y[:id], label="MPC trajectory", color="green")
    ax_traj.plot(X[id - 1 :], Y[id - 1 :], label="DPC trajectory", color="red")
    ax_traj.scatter([0.0], [0.0], marker="x", color="green")
    ax_traj.scatter(
        track.blue_cones[:, 0], track.blue_cones[:, 1], color="blue", s=15, marker="^"
    )
    ax_traj.scatter(
        track.yellow_cones[:, 0],
        track.yellow_cones[:, 1],
        color="yellow",
        s=15,
        marker="^",
    )
    ax_traj.scatter(
        track.big_orange_cones[:, 0],
        track.big_orange_cones[:, 1],
        color="orange",
        s=30,
        marker="^",
    )
    ax_traj.legend(loc=3)
    ax_traj.set_xlim(np.min(X) - 2, np.max(X) + 2)
    ax_traj.set_ylim(np.min(Y) - 1, np.max(Y) + 3)
    ax_traj.set_aspect("equal")

    ax_T.plot(np.arange(id) * 0.05, T[:id], color="green")
    ax_T.plot(np.arange(id - 1, T.shape[0]) * 0.05, T[id - 1 :], color="red")
    ax_T.set_ylabel(r"$T$ [1]")
    ax_delta.plot(np.arange(id) * 0.05, delta[:id], color="green")
    ax_delta.plot(
        np.arange(id - 1, delta.shape[0]) * 0.05, delta[id - 1 :], color="red"
    )
    ax_delta.set_ylabel(r"$\delta$ [°]")
    ax_ddelta.plot(np.arange(id) * 0.05, ddelta[:id], color="green")
    ax_ddelta.plot(
        np.arange(id - 1, ddelta.shape[0]) * 0.05, ddelta[id - 1 :], color="red"
    )
    ax_ddelta.set_ylabel(r"$d\delta$ [°/s]")
    ax_ddelta.set_xlabel("simulation time [s]")

    fig.tight_layout()
    plt.savefig("experiments/control/dpc_cl.png", dpi=300)
    # plt.show()


if __name__ == "__main__":
    print("Open loop ======================================== ")
    control_open_loop()
    print("Closed loop ========================================")
    control_closed_loop()
