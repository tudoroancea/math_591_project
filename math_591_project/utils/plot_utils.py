# Copyright (c) 2023 Tudor Oancea
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from data_visualization import *
from icecream import ic

if sys.platform == "linux":
    matplotlib.use("TkAgg")

__all__ = ["plot_sysid_trajs", "plot_control_trajs"]


def plot_sysid_trajs(
    xtilde0: np.ndarray,
    utilde0toNfminus1: np.ndarray,
    xtilde1toNf: np.ndarray,
    xtilde1toNf_p: np.ndarray,
    model_labels: list[str],
    dt=1 / 20,
):
    """
    Plot the trajectories of the system identification for a bunch of models (Nmodels).
    If one of the models is kin4 or blackbox_kin4, it just has to use nans for the last
    two states (v_y and r).

    :param xtilde0: initial state, shape (1, 6)
    :param utilde0toNfminus1: control inputs, shape (Nf, 2)
    :param xtilde1toNf: true state trajectories, shape (Nf, 6)
    :param xtilde1toNf_p: predicted state trajectories, shape (Nmodels, Nf, 6)
    """
    assert (
        xtilde1toNf.shape[0] == utilde0toNfminus1.shape[0] == xtilde1toNf_p.shape[1]
    ), (
        "dimension mismatch: "
        f"xtilde1toNf.shape[0] = {xtilde1toNf.shape[0]}, "
        f"utilde0toNfminus1.shape[0] = {utilde0toNfminus1.shape[0]}, "
        f"xtilde1toNf_p.shape[1] = {xtilde1toNf_p.shape[1]}"
    )
    Nf = xtilde1toNf.shape[0]
    Nmodels = xtilde1toNf_p.shape[0]
    xtilde1toNf = np.concatenate((xtilde0.reshape(1, -1), xtilde1toNf), axis=0)
    xtilde1toNf_p = np.concatenate(
        (np.tile(xtilde0.reshape(1, -1), (Nmodels, 1, 1)), xtilde1toNf_p), axis=1
    )
    utilde0toNfminus1 = np.concatenate(
        (utilde0toNfminus1, np.full((1, utilde0toNfminus1.shape[1]), np.nan)), axis=0
    )
    colors = ["blue", "orange", "green", "red", "purple", "brown"]
    linewidth = 1.5
    simulation_plot = Plot(
        row_nbr=3,
        col_nbr=3,
        mode=PlotMode.STATIC,
        sampling_time=dt,
        interval=1,
        figsize=(15, 8),
    )
    traj_data = {
        "true trajectory": {
            "data": xtilde1toNf[:, :2],
            "curve_type": CurveType.REGULAR,
            "curve_style": CurvePlotStyle.PLOT,
            "mpl_options": {"color": "black", "linewidth": linewidth},
        }
    }
    for i, model_label in enumerate(model_labels):
        traj_data["predicted trajectory " + model_label] = {
            "data": xtilde1toNf_p[i, :, :2],
            "curve_type": CurveType.REGULAR,
            "curve_style": CurvePlotStyle.PLOT,
            "mpl_options": {"color": colors[i], "linewidth": linewidth},
        }
    simulation_plot.add_subplot(
        row_idx=range(3),
        col_idx=0,
        subplot_name=r"$XY$",
        subplot_type=SubplotType.SPATIAL,
        unit="m",
        show_unit=True,
        curves=traj_data,
    )
    phi_data = {
        r"true $\phi$": {
            "data": np.rad2deg(xtilde1toNf[:, 2]),
            "curve_type": CurveType.REGULAR,
            "curve_style": CurvePlotStyle.PLOT,
            "mpl_options": {"color": "black", "linewidth": linewidth},
        }
    }
    for i, model_label in enumerate(model_labels):
        phi_data[r"predicted $\varphi$ " + model_label] = {
            "data": np.rad2deg(xtilde1toNf_p[i, :, 2]),
            "curve_type": CurveType.REGULAR,
            "curve_style": CurvePlotStyle.PLOT,
            "mpl_options": {"color": colors[i], "linewidth": linewidth},
        }
    simulation_plot.add_subplot(
        row_idx=0,
        col_idx=1,
        subplot_name=r"$\varphi$",
        subplot_type=SubplotType.TEMPORAL,
        unit="°",
        show_unit=True,
        curves=phi_data,
    )
    v_x_data = {
        r"true $v_x$": {
            "data": xtilde1toNf[:, 3],
            "curve_type": CurveType.REGULAR,
            "curve_style": CurvePlotStyle.PLOT,
            "mpl_options": {"color": "black", "linewidth": linewidth},
        }
    }
    for i, model_label in enumerate(model_labels):
        v_x_data[r"predicted $v_x$ " + model_label] = {
            "data": xtilde1toNf_p[i, :, 3],
            "curve_type": CurveType.REGULAR,
            "curve_style": CurvePlotStyle.PLOT,
            "mpl_options": {"color": colors[i], "linewidth": linewidth},
        }
    simulation_plot.add_subplot(
        row_idx=1,
        col_idx=1,
        subplot_name=r"$v_x$",
        subplot_type=SubplotType.TEMPORAL,
        unit="m/s",
        show_unit=True,
        curves=v_x_data,
    )

    simulation_plot.add_subplot(
        row_idx=2,
        col_idx=1,
        subplot_name=r"$T$",
        subplot_type=SubplotType.TEMPORAL,
        unit="1",
        show_unit=True,
        curves={
            "T": {
                "data": utilde0toNfminus1[:, 0],
                "curve_type": CurveType.REGULAR,
                "curve_style": CurvePlotStyle.STEP,
                "mpl_options": {
                    "color": "black",
                    "linewidth": linewidth,
                    "where": "post",
                },
            },
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
            "delta": {
                "data": np.rad2deg(utilde0toNfminus1[:, 1]),
                "curve_type": CurveType.REGULAR,
                "curve_style": CurvePlotStyle.STEP,
                "mpl_options": {
                    "color": "black",
                    "linewidth": linewidth,
                    "where": "post",
                },
            },
        },
    )
    r_data = {
        r"true $r$": {
            "data": np.rad2deg(xtilde1toNf[:, 5]),
            "curve_type": CurveType.REGULAR,
            "curve_style": CurvePlotStyle.PLOT,
            "mpl_options": {"color": "black", "linewidth": linewidth},
        }
    }
    for i, model_label in enumerate(model_labels):
        r_data[r"predicted $r$ " + model_label] = {
            "data": np.rad2deg(xtilde1toNf_p[i, :, 5]),
            "curve_type": CurveType.REGULAR,
            "curve_style": CurvePlotStyle.PLOT,
            "mpl_options": {"color": colors[i], "linewidth": linewidth},
        }
    simulation_plot.add_subplot(
        row_idx=0,
        col_idx=2,
        subplot_name=r"$r$",
        subplot_type=SubplotType.TEMPORAL,
        unit="°/s",
        show_unit=True,
        curves=r_data,
    )
    v_y_data = {
        r"true $v_y$": {
            "data": xtilde1toNf[:, 4],
            "curve_type": CurveType.REGULAR,
            "curve_style": CurvePlotStyle.PLOT,
            "mpl_options": {"color": "black", "linewidth": linewidth},
        }
    }
    for i, model_label in enumerate(model_labels):
        v_y_data[r"predicted $v_y$ " + model_label] = {
            "data": xtilde1toNf_p[i, :, 4],
            "curve_type": CurveType.REGULAR,
            "curve_style": CurvePlotStyle.PLOT,
            "mpl_options": {"color": colors[i], "linewidth": linewidth},
        }
    simulation_plot.add_subplot(
        row_idx=1,
        col_idx=2,
        subplot_name=r"$v_y$",
        subplot_type=SubplotType.TEMPORAL,
        unit="m/s",
        show_unit=True,
        curves=v_y_data,
    )
    simulation_plot.plot(show=False)
    simulation_plot._content[r"$XY$"]["ax"].legend(
        ["ground truth"] + model_labels, loc=2
    )


def plot_control_trajs(
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

    :param x0: initial state, shape (1, 7)
    :param xref0toNf: reference trajectory, shape (Nf+1, 4)
    :param u0toNfminus1: computed controls, shape (Nobs, Nf, 2)
    :param x1toNf: open loop state prediction, shape (Nobs, Nf, 7)
    """
    assert x0.size == x1toNf.shape[2] == 7
    assert xref0toNf.shape[1] == 4
    assert u0toNfminus1.shape[2] == 2
    assert (xref0toNf.shape[0] - 1) == u0toNfminus1.shape[1] == x1toNf.shape[1]
    assert x1toNf.shape[0] == u0toNfminus1.shape[0] == len(model_labels)
    Nobs = x1toNf.shape[0]
    Nf = x1toNf.shape[1]

    x0 = x0.ravel()
    X = np.concatenate((np.full((Nobs, 1), x0[0]), x1toNf[..., 0]), axis=1)
    Y = np.concatenate((np.full((Nobs, 1), x0[1]), x1toNf[..., 1]), axis=1)
    phi = np.concatenate((np.full((Nobs, 1), x0[2]), x1toNf[..., 2]), axis=1)
    v_x = np.concatenate((np.full((Nobs, 1), x0[3]), x1toNf[..., 3]), axis=1)
    v_y = np.concatenate((np.full((Nobs, 1), x0[4]), x1toNf[..., 4]), axis=1)
    r = np.concatenate((np.full((Nobs, 1), x0[5]), x1toNf[..., 5]), axis=1)
    T = np.concatenate((u0toNfminus1[..., 0], np.full((Nobs, 1), np.nan)), axis=1)
    delta = np.concatenate((x1toNf[:, :, 6], np.full((Nobs, 1), np.nan)), axis=1)
    ddelta = np.concatenate((u0toNfminus1[..., 1], np.full((Nobs, 1), np.nan)), axis=1)

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
            for i in range(Nobs)
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
            for i in range(Nobs)
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
            for i in range(Nobs)
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
            for i in range(Nobs)
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
            for i in range(Nobs)
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
            for i in range(Nobs)
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
            for i in range(Nobs)
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
            for i in range(Nobs)
        },
    )
    simulation_plot.plot(show=False)
    simulation_plot._content[r"$XY$"]["ax"].legend(["reference"] + model_labels, loc=2)
