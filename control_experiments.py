# Copyright (c) 2023 Tudor Oancea
import argparse
import json
import os

import lightning as L
import torch
import torch.nn.functional as F
from lightning import Fabric
from matplotlib import pyplot as plt

from math_591_project.utils.data_utils import *
from math_591_project.models import *
from math_591_project.utils.plot_utils import *
from data_visualization import *
from track_database import *

L.seed_everything(127)
plt.style.use(["science"])
plt.rcParams.update({"font.size": 20})


def load_config(config_path):
    return json.load(open(config_path, "r"))


def load_control_test_data(fabric: Fabric, config: dict):
    data_dir = os.path.join(config["data"]["dir"], config["data"]["test"])
    file_paths = [
        os.path.abspath(os.path.join(data_dir, file_path))
        for file_path in os.listdir(data_dir)
        if file_path.endswith(".csv")
    ]
    test_dataset = ControlDataset(file_paths)
    return test_dataset
    # test_dataloader = DataLoader(
    #     test_dataset, batch_size=1, shuffle=True, num_workers=1
    # )
    # test_dataloader = fabric.setup_dataloaders(test_dataloader)

    # return test_dataset, test_dataloader


def create_control_test_model(fabric: Fabric, config: dict):
    model_name = config["model"]["name"]
    is_blackbox = model_name.startswith("blackbox")
    dims = ode_dims[model_name]
    if is_blackbox:
        ode_t, nxtilde, nutilde, nin, nout = dims
    else:
        ode_t, nxtilde, nutilde = dims

    system_model = OpenLoop(
        model=RK4(
            nxtilde=nxtilde,
            nutilde=nutilde,
            ode=ode_t(
                net=MLP(
                    nin=nin,
                    nout=nout,
                    nhidden=config["model"]["n_hidden"],
                    nonlinearity=config["model"]["nonlinearity"],
                )
            )
            if is_blackbox
            else ode_t(),
            dt=1 / 20,
        ),
        Nf=config["testing"]["Nf"],
    )
    try:
        system_model.model.ode.load_state_dict(
            torch.load(config["model"]["checkpoint_path"], map_location="cpu")[
                "system_model"
            ]
        )
        # print("Successfully loaded model parameters from checkpoint")
    except FileNotFoundError:
        print("No checkpoint found, using random initialization")
    except RuntimeError:
        print("Checkpoint found, but not compatible with current model")

    system_model = fabric.setup(system_model)

    return system_model


def plot_stuff(
    x0: np.ndarray,
    xref0toNf: np.ndarray,
    u0toNfminus1: np.ndarray,
    x1toNf: np.ndarray,
    model_labels: list[str],
    dt=1 / 20,
):
    """
    Plot the trajectories of the system identification for a bunch of models (Nmodels).
    If one of the models is kin4 or blackbox_kin4, it just has to use nans for the last
    two states (v_y and r).

    :param x0: initial state, shape (1, 6)
    :param xref0toNf: reference trajectory, shape (Nf, 6)
    :param u0toNfminus1: computed controls, shape (M, Nf, 2)
    :param x1toNf: open loop state prediction, shape (M, Nf, 6)
    """
    assert len(u0toNfminus1.shape) == 3
    M = u0toNfminus1.shape[0]
    assert len(model_labels) == M == x1toNf.shape[0]
    # x0toNf = np.concatenate((np.tile(x0.reshape(1, -1), (M, 1, 1)), x1toNf), axis=1)
    x0 = x0.ravel()
    X = np.concatenate((np.full((M, 1), x0[0]), x1toNf[..., 0]), axis=1)
    Y = np.concatenate((np.full((M, 1), x0[1]), x1toNf[..., 1]), axis=1)
    phi = np.concatenate((np.full((M, 1), x0[2]), x1toNf[..., 2]), axis=1)
    v_x = np.concatenate((np.full((M, 1), x0[3]), x1toNf[..., 3]), axis=1)
    v_y = np.concatenate((np.full((M, 1), x0[4]), x1toNf[..., 4]), axis=1)
    r = np.concatenate((np.full((M, 1), x0[5]), x1toNf[..., 5]), axis=1)
    T = np.concatenate((u0toNfminus1[..., 0], np.full((M, 1), np.nan)), axis=1)
    delta = np.concatenate((x1toNf[:, :, 6], np.full((M, 1), np.nan)), axis=1)
    ddelta = np.concatenate((u0toNfminus1[..., 1], np.full((M, 1), np.nan)), axis=1)

    # ref_options = {"color": "blue", "linewidth": 2}
    # pred_options = {"color": "red", "linewidth": 2}
    colors = ["blue", "orange", "green", "red", "purple", "brown"]
    linewidth = 1.5
    simulation_plot = Plot(
        row_nbr=4,
        col_nbr=3,
        mode=PlotMode.STATIC,
        sampling_time=dt,
        interval=1,
        figsize=(15, 8),
    )
    simulation_plot.add_subplot(
        row_idx=range(4),
        col_idx=0,
        subplot_name=r"$XY$",
        subplot_type=SubplotType.SPATIAL,
        unit="m",
        show_unit=True,
        curves={
            "reference trajectory": {
                "data": xref0toNf[:, :2],
                "curve_type": CurveType.REGULAR,
                "curve_style": CurvePlotStyle.PLOT,
                "mpl_options": {"color": "black", "linewidth": linewidth},
            }
        }
        | {
            model_labels[i]: {
                "data": np.array([X[i], Y[i]]).T,
                "curve_type": CurveType.REGULAR,
                "curve_style": CurvePlotStyle.PLOT,
                "mpl_options": {"color": colors[i], "linewidth": linewidth},
            }
            for i in range(M)
        },
    )
    simulation_plot.add_subplot(
        row_idx=0,
        col_idx=1,
        subplot_name=r"$\varphi$",
        subplot_type=SubplotType.TEMPORAL,
        unit="°",
        show_unit=True,
        curves={
            r"reference $\phi$": {
                "data": np.rad2deg(xref0toNf[:, 2]),
                "curve_type": CurveType.REGULAR,
                "curve_style": CurvePlotStyle.PLOT,
                "mpl_options": {"color": "black", "linewidth": linewidth},
            }
        }
        | {
            model_labels[i]: {
                "data": np.rad2deg(phi[i]),
                "curve_type": CurveType.REGULAR,
                "curve_style": CurvePlotStyle.PLOT,
                "mpl_options": {"color": colors[i], "linewidth": linewidth},
            }
            for i in range(M)
        },
    )
    simulation_plot.add_subplot(
        row_idx=1,
        col_idx=1,
        subplot_name=r"$v_x$",
        subplot_type=SubplotType.TEMPORAL,
        unit="m/s",
        show_unit=True,
        curves={
            r"reference $v_x$": {
                "data": xref0toNf[:, 3],
                "curve_type": CurveType.REGULAR,
                "curve_style": CurvePlotStyle.PLOT,
                "mpl_options": {"color": "black", "linewidth": linewidth},
            },
        }
        | {
            model_labels[i]: {
                "data": v_x[i],
                "curve_type": CurveType.REGULAR,
                "curve_style": CurvePlotStyle.PLOT,
                "mpl_options": {"color": colors[i], "linewidth": linewidth},
            }
            for i in range(M)
        },
    )
    simulation_plot.add_subplot(
        row_idx=2,
        col_idx=1,
        subplot_name=r"$T$",
        subplot_type=SubplotType.TEMPORAL,
        unit="1",
        show_unit=True,
        curves={
            model_labels[i]: {
                "data": T[i],
                "curve_type": CurveType.REGULAR,
                "curve_style": CurvePlotStyle.STEP,
                "mpl_options": {
                    "color": colors[i],
                    "linewidth": linewidth,
                    "where": "post",
                },
            }
            for i in range(M)
        },
    )
    simulation_plot.add_subplot(
        row_idx=2,
        col_idx=2,
        subplot_name=r"$\delta$",
        subplot_type=SubplotType.TEMPORAL,
        unit="°",
        show_unit=True,
        curves={
            model_labels[i]: {
                "data": np.rad2deg(delta[i]),
                "curve_type": CurveType.REGULAR,
                "curve_style": CurvePlotStyle.STEP,
                "mpl_options": {
                    "color": colors[i],
                    "linewidth": linewidth,
                    "where": "post",
                },
            }
            for i in range(M)
        },
    )
    simulation_plot.add_subplot(
        row_idx=0,
        col_idx=2,
        subplot_name=r"$r$",
        subplot_type=SubplotType.TEMPORAL,
        unit="°/s",
        show_unit=True,
        curves={
            model_labels[i]: {
                "data": np.rad2deg(r[i]),
                "curve_type": CurveType.REGULAR,
                "curve_style": CurvePlotStyle.PLOT,
                "mpl_options": {
                    "color": colors[i],
                    "linewidth": linewidth,
                },
            }
            for i in range(M)
        },
    )
    simulation_plot.add_subplot(
        row_idx=1,
        col_idx=2,
        subplot_name=r"$v_y$",
        subplot_type=SubplotType.TEMPORAL,
        unit="m/s",
        show_unit=True,
        curves={
            model_labels[i]: {
                "data": v_y[i],
                "curve_type": CurveType.REGULAR,
                "curve_style": CurvePlotStyle.PLOT,
                "mpl_options": {
                    "color": colors[i],
                    "linewidth": linewidth,
                },
            }
            for i in range(M)
        },
    )
    simulation_plot.add_subplot(
        row_idx=3,
        col_idx=2,
        subplot_name=r"$d\delta$",
        subplot_type=SubplotType.TEMPORAL,
        unit="°/s",
        show_unit=True,
        curves={
            model_labels[i]: {
                "data": np.rad2deg(ddelta[i]),
                "curve_type": CurveType.REGULAR,
                "curve_style": CurvePlotStyle.STEP,
                "mpl_options": {
                    "color": colors[i],
                    "linewidth": linewidth,
                    "where": "post",
                },
            }
            for i in range(M)
        },
    )
    simulation_plot.plot(show=False)
    simulation_plot._content[r"$XY$"]["ax"].legend(["reference"] + model_labels, loc=2)


@torch.no_grad()
def control_exp1():
    config_path = "config/blackbox_dyn6_control_dpc.json"
    config = load_config(config_path)
    # load control dataset
    fabric = Fabric()
    test_dataset = load_control_test_data(fabric, config)
    # load system model
    dt = 1 / 20
    Nf = 40
    kin4_model = OpenLoop(
        model=LiftedDiscreteModel(
            nx=KIN4_NX,
            nu=KIN4_NUTILDE,
            model=RK4(
                nxtilde=KIN4_NXTILDE,
                nutilde=KIN4_NUTILDE,
                ode=Kin4ODE(),
                dt=dt,
            ),
        ),
        Nf=Nf,
    )
    blackbox_dyn6_model = OpenLoop(
        model=LiftedDiscreteModel(
            nx=DYN6_NX,
            nu=DYN6_NUTILDE,
            model=RK4(
                nxtilde=DYN6_NXTILDE,
                nutilde=DYN6_NUTILDE,
                ode=BlackboxDyn6ODE(
                    net=MLP(
                        nin=ode_dims["blackbox_dyn6"][3],
                        nout=ode_dims["blackbox_dyn6"][4],
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
        nx=DYN6_NX,
        nu=DYN6_NUTILDE,
        Nf=Nf,
        mlp=MLP(
            nin=DYN6_NX - 3 + (Nf + 1) * 4,
            nout=DYN6_NUTILDE * Nf,
            nhidden=config["control_model"]["nhidden"],
            nonlinearity=config["control_model"]["nonlinearity"],
        ),
    )
    try:
        blackbox_dyn6_model.model.model.ode.load_state_dict(
            torch.load(config["system_model"]["checkpoint_path"], map_location="cpu")[
                "system_model"
            ]
        )
        control_model.mlp.load_state_dict(
            torch.load(config["control_model"]["checkpoint_path"], map_location="cpu")[
                "control_model"
            ]
        )
    except FileNotFoundError:
        print("No checkpoint found, using random initialization")
    except RuntimeError:
        print("Checkpoint found, but not compatible with current model")

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
    #  4. dpc with kin4_model
    # x1toNf = kin4_model(x0[..., [0, 1, 2, 3, 6]], u0toNfminus1)
    # x1toNf = torch.cat(
    #     (
    #         x1toNf[..., [0, 1, 2, 3]],
    #         torch.full((1, Nf, 2), torch.nan, device=x1toNf.device),
    #         x1toNf[..., [4]],
    #     ),
    #     dim=-1,
    # )
    # plot_x1toNf.append(x1toNf.cpu().numpy())
    # plot_u0toNf.append(u0toNfminus1.cpu().numpy())

    # plot
    plot_u0toNf = np.concatenate(plot_u0toNf, axis=0)
    plot_x1toNf = np.concatenate(plot_x1toNf, axis=0)
    x0 = x0.squeeze(0).cpu().numpy()
    xref0toNf = xref0toNf.squeeze(0).cpu().numpy()
    plot_stuff(
        x0,
        xref0toNf,
        plot_u0toNf,
        plot_x1toNf,
        [
            "MPC+NeuralDyn6",
            "MPC+Kin4",
            "DPC+NeuralDyn6",
            # "DPC+Kin4",
        ],
    )
    plt.savefig("dpc_ol.png", dpi=300)
    # plt.show()


def control_exp2():
    df = pd.read_csv("dpc_w_warmstarting.csv")
    method = df["method"].values
    X = df["X"].values
    Y = df["Y"].values
    # phi = df["phi"].values
    # v_x = df["v_x"].values
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
    plt.savefig("dpc_cl.png", dpi=300)
    # plt.show()


def control_dataset_size():
    train_data_dir = "data_v1.1.0/train"
    train_data_dir = "data_v1.1.0/test"
    file_paths = [
        os.path.abspath(os.path.join(train_data_dir, file_path))
        for file_path in os.listdir(train_data_dir)
        if file_path.endswith(".csv")
    ]
    control_dataset = ControlDataset(file_paths)
    print(len(control_dataset))


if __name__ == "__main__":
    # main()
    # plot_errors()
    # dataset_velocity_distribution()
    # plot_losses()
    # plot_trajs()
    control_exp1()
    # control_exp2()
    # control_dataset_size()
