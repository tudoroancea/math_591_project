# Copyright (c) 2023 Tudor Oancea
import sys

import matplotlib
from matplotlib.widgets import EllipseSelector
import numpy as np
from data_visualization import *

if sys.platform == "linux":
    matplotlib.use("TkAgg")

__all__ = ["plot_kin4", "plot_dyn6", "plot_kin4_control", "plot_dyn6_control"]


def plot_kin4(
    xtilde0: np.ndarray,
    utilde0toNfminus1: np.ndarray,
    xtilde1toNf: np.ndarray,
    xtilde1toNf_p: np.ndarray,
    dt=1 / 20,
):
    xtilde1toNf = np.concatenate((xtilde0.reshape(1, -1), xtilde1toNf), axis=0)
    xtilde1toNf_p = np.concatenate((xtilde0.reshape(1, -1), xtilde1toNf_p), axis=0)
    utilde0toNfminus1 = np.concatenate(
        (utilde0toNfminus1, np.full((1, utilde0toNfminus1.shape[1])), np.nan), axis=0
    )

    simulation_plot = Plot(
        row_nbr=2,
        col_nbr=3,
        mode=PlotMode.STATIC,
        sampling_time=dt,
        interval=1,
        figsize=(15, 8),
    )
    simulation_plot.add_subplot(
        row_idx=range(2),
        col_idx=0,
        subplot_name="map",
        subplot_type=SubplotType.SPATIAL,
        unit="m",
        show_unit=True,
        curves={
            "true trajectory": {
                "data": xtilde1toNf[:, :2],
                "curve_type": CurveType.REGULAR,
                "curve_style": CurvePlotStyle.PLOT,
                "mpl_options": {"color": "blue", "linewidth": 2},
            },
            "predicted trajectory": {
                "data": xtilde1toNf_p[:, :2],
                "curve_type": CurveType.REGULAR,
                "curve_style": CurvePlotStyle.PLOT,
                "mpl_options": {"color": "red", "linewidth": 2},
            },
        },
    )
    simulation_plot.add_subplot(
        row_idx=0,
        col_idx=1,
        subplot_name="phi",
        subplot_type=SubplotType.TEMPORAL,
        unit="°",
        show_unit=True,
        curves={
            r"true $\phi$": {
                "data": np.rad2deg(xtilde1toNf[:, 2]),
                "curve_type": CurveType.REGULAR,
                "curve_style": CurvePlotStyle.PLOT,
                "mpl_options": {"color": "blue", "linewidth": 2},
            },
            r"predicted $\phi$": {
                "data": np.rad2deg(xtilde1toNf_p[:, 2]),
                "curve_type": CurveType.REGULAR,
                "curve_style": CurvePlotStyle.PLOT,
                "mpl_options": {"color": "red", "linewidth": 2},
            },
        },
    )
    simulation_plot.add_subplot(
        row_idx=1,
        col_idx=1,
        subplot_name="v_x",
        subplot_type=SubplotType.TEMPORAL,
        unit="m/s",
        show_unit=True,
        curves={
            r"true $v_x$": {
                "data": xtilde1toNf[:, 3],
                "curve_type": CurveType.REGULAR,
                "curve_style": CurvePlotStyle.PLOT,
                "mpl_options": {"color": "blue", "linewidth": 2},
            },
            "predicted $v_x$": {
                "data": xtilde1toNf_p[:, 3],
                "curve_type": CurveType.REGULAR,
                "curve_style": CurvePlotStyle.PLOT,
                "mpl_options": {"color": "red", "linewidth": 2},
            },
        },
    )
    simulation_plot.add_subplot(
        row_idx=0,
        col_idx=2,
        subplot_name="delta",
        subplot_type=SubplotType.TEMPORAL,
        unit="°",
        show_unit=True,
        curves={
            "delta": {
                "data": np.rad2deg(utilde0toNfminus1[:, 1]),
                "curve_type": CurveType.REGULAR,
                "curve_style": CurvePlotStyle.STEP,
                "mpl_options": {"color": "blue", "linewidth": 2, "where": "post"},
            },
        },
    )
    simulation_plot.add_subplot(
        row_idx=1,
        col_idx=2,
        subplot_name="T",
        subplot_type=SubplotType.TEMPORAL,
        unit="1",
        show_unit=False,
        curves={
            "T": {
                "data": utilde0toNfminus1[:, 0],
                "curve_type": CurveType.REGULAR,
                "curve_style": CurvePlotStyle.STEP,
                "mpl_options": {"color": "blue", "linewidth": 2, "where": "post"},
            },
        },
    )
    simulation_plot.plot(show=False)


def plot_dyn6(
    xtilde0: np.ndarray,
    utilde0toNfminus1: np.ndarray,
    xtilde1toNf: np.ndarray,
    xtilde1toNf_p: np.ndarray,
    dt=1 / 20,
):
    xtilde1toNf = np.concatenate((xtilde0.reshape(1, -1), xtilde1toNf), axis=0)
    xtilde1toNf_p = np.concatenate((xtilde0.reshape(1, -1), xtilde1toNf_p), axis=0)
    utilde0toNfminus1 = np.concatenate(
        (utilde0toNfminus1, np.full((1, utilde0toNfminus1.shape[1]), np.nan)), axis=0
    )

    simulation_plot = Plot(
        row_nbr=3,
        col_nbr=3,
        mode=PlotMode.STATIC,
        sampling_time=dt,
        interval=1,
        figsize=(15, 8),
    )
    simulation_plot.add_subplot(
        row_idx=range(3),
        col_idx=0,
        subplot_name="map",
        subplot_type=SubplotType.SPATIAL,
        unit="m",
        show_unit=True,
        curves={
            "true trajectory": {
                "data": xtilde1toNf[:, :2],
                "curve_type": CurveType.REGULAR,
                "curve_style": CurvePlotStyle.PLOT,
                "mpl_options": {"color": "blue", "linewidth": 2},
            },
            "predicted trajectory": {
                "data": xtilde1toNf_p[:, :2],
                "curve_type": CurveType.REGULAR,
                "curve_style": CurvePlotStyle.PLOT,
                "mpl_options": {"color": "red", "linewidth": 2},
            },
        },
    )
    simulation_plot.add_subplot(
        row_idx=0,
        col_idx=1,
        subplot_name="phi",
        subplot_type=SubplotType.TEMPORAL,
        unit="°",
        show_unit=True,
        curves={
            r"true $\phi$": {
                "data": np.rad2deg(xtilde1toNf[:, 2]),
                "curve_type": CurveType.REGULAR,
                "curve_style": CurvePlotStyle.PLOT,
                "mpl_options": {"color": "blue", "linewidth": 2},
            },
            r"predicted $\phi$": {
                "data": np.rad2deg(xtilde1toNf_p[:, 2]),
                "curve_type": CurveType.REGULAR,
                "curve_style": CurvePlotStyle.PLOT,
                "mpl_options": {"color": "red", "linewidth": 2},
            },
        },
    )
    simulation_plot.add_subplot(
        row_idx=1,
        col_idx=1,
        subplot_name="v_x",
        subplot_type=SubplotType.TEMPORAL,
        unit="m/s",
        show_unit=True,
        curves={
            r"true $v_x$": {
                "data": xtilde1toNf[:, 3],
                "curve_type": CurveType.REGULAR,
                "curve_style": CurvePlotStyle.PLOT,
                "mpl_options": {"color": "blue", "linewidth": 2},
            },
            "predicted $v_x$": {
                "data": xtilde1toNf_p[:, 3],
                "curve_type": CurveType.REGULAR,
                "curve_style": CurvePlotStyle.PLOT,
                "mpl_options": {"color": "red", "linewidth": 2},
            },
        },
    )
    simulation_plot.add_subplot(
        row_idx=2,
        col_idx=1,
        subplot_name="T",
        subplot_type=SubplotType.TEMPORAL,
        unit="1",
        show_unit=False,
        curves={
            "T": {
                "data": utilde0toNfminus1[:, 0],
                "curve_type": CurveType.REGULAR,
                "curve_style": CurvePlotStyle.STEP,
                "mpl_options": {"color": "blue", "linewidth": 2, "where": "post"},
            },
        },
    )
    simulation_plot.add_subplot(
        row_idx=2,
        col_idx=2,
        subplot_name="delta",
        subplot_type=SubplotType.TEMPORAL,
        unit="°",
        show_unit=True,
        curves={
            "delta": {
                "data": np.rad2deg(utilde0toNfminus1[:, 1]),
                "curve_type": CurveType.REGULAR,
                "curve_style": CurvePlotStyle.STEP,
                "mpl_options": {"color": "blue", "linewidth": 2, "where": "post"},
            },
        },
    )
    simulation_plot.add_subplot(
        row_idx=0,
        col_idx=2,
        subplot_name="r",
        subplot_type=SubplotType.TEMPORAL,
        unit="°/s",
        show_unit=True,
        curves={
            "true r": {
                "data": np.rad2deg(xtilde1toNf[:, 5]),
                "curve_type": CurveType.REGULAR,
                "curve_style": CurvePlotStyle.PLOT,
                "mpl_options": {"color": "blue", "linewidth": 2},
            },
            "predicted r": {
                "data": np.rad2deg(xtilde1toNf_p[:, 5]),
                "curve_type": CurveType.REGULAR,
                "curve_style": CurvePlotStyle.PLOT,
                "mpl_options": {"color": "red", "linewidth": 2},
            },
        },
    )
    simulation_plot.add_subplot(
        row_idx=1,
        col_idx=2,
        subplot_name="v_y",
        subplot_type=SubplotType.TEMPORAL,
        unit="m/s",
        show_unit=True,
        curves={
            "true v_y": {
                "data": xtilde1toNf[:, 4],
                "curve_type": CurveType.REGULAR,
                "curve_style": CurvePlotStyle.PLOT,
                "mpl_options": {"color": "blue", "linewidth": 2},
            },
            "predicted v_y": {
                "data": xtilde1toNf_p[:, 4],
                "curve_type": CurveType.REGULAR,
                "curve_style": CurvePlotStyle.PLOT,
                "mpl_options": {"color": "red", "linewidth": 2},
            },
        },
    )
    simulation_plot.plot(show=False)


def plot_kin4_control(
    x0: np.ndarray,
    xref0toNf: np.ndarray,
    u0toNfminus1: np.ndarray,
    x1toNf: np.ndarray,
    dt=1 / 20,
):
    pass


def plot_dyn6_control(
    x0: np.ndarray,
    xref0toNf: np.ndarray,
    u0toNfminus1: np.ndarray,
    x1toNf: np.ndarray,
    uref0toNfminus1: np.ndarray = None,
    dt=1 / 20,
):
    x0toNf = np.concatenate((x0.reshape(1, -1), x1toNf), axis=0)
    T = np.append(u0toNfminus1[:, 0], np.nan)
    delta = np.append(x1toNf[:, -1], np.nan)
    ddelta = np.append(u0toNfminus1[:, 1], np.nan)
    if uref0toNfminus1 is not None:
        T_ref = np.append(uref0toNfminus1[:, 0], np.nan)
        ddelta_ref = np.append(uref0toNfminus1[:, 1], np.nan)
    else:
        T_ref = np.full_like(T, np.nan)
        ddelta_ref = np.full_like(ddelta, np.nan)

    ref_options = {"color": "blue", "linewidth": 2}
    pred_options = {"color": "red", "linewidth": 2}
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
        subplot_name="map",
        subplot_type=SubplotType.SPATIAL,
        unit="m",
        show_unit=True,
        curves={
            "reference trajectory": {
                "data": xref0toNf[:, :2],
                "curve_type": CurveType.REGULAR,
                "curve_style": CurvePlotStyle.PLOT,
                "mpl_options": ref_options,
            },
            "predicted trajectory": {
                "data": x0toNf[:, :2],
                "curve_type": CurveType.REGULAR,
                "curve_style": CurvePlotStyle.PLOT,
                "mpl_options": pred_options,
            },
        },
    )
    simulation_plot.add_subplot(
        row_idx=0,
        col_idx=1,
        subplot_name="phi",
        subplot_type=SubplotType.TEMPORAL,
        unit="°",
        show_unit=True,
        curves={
            r"reference $\phi$": {
                "data": np.rad2deg(xref0toNf[:, 2]),
                "curve_type": CurveType.REGULAR,
                "curve_style": CurvePlotStyle.PLOT,
                "mpl_options": ref_options,
            },
            r"predicted $\phi$": {
                "data": np.rad2deg(x0toNf[:, 2]),
                "curve_type": CurveType.REGULAR,
                "curve_style": CurvePlotStyle.PLOT,
                "mpl_options": pred_options,
            },
        },
    )
    simulation_plot.add_subplot(
        row_idx=1,
        col_idx=1,
        subplot_name="v_x",
        subplot_type=SubplotType.TEMPORAL,
        unit="m/s",
        show_unit=True,
        curves={
            r"reference $v_x$": {
                "data": xref0toNf[:, 3],
                "curve_type": CurveType.REGULAR,
                "curve_style": CurvePlotStyle.PLOT,
                "mpl_options": ref_options,
            },
            r"predicted $v_x$": {
                "data": x0toNf[:, 3],
                "curve_type": CurveType.REGULAR,
                "curve_style": CurvePlotStyle.PLOT,
                "mpl_options": pred_options,
            },
        },
    )
    simulation_plot.add_subplot(
        row_idx=2,
        col_idx=1,
        subplot_name="T",
        subplot_type=SubplotType.TEMPORAL,
        unit="1",
        show_unit=False,
        curves={
            r"$T$": {
                "data": T,
                "curve_type": CurveType.REGULAR,
                "curve_style": CurvePlotStyle.STEP,
                "mpl_options": pred_options | {"where": "post"},
            },
            r"$T_{ref}$": {
                "data": T_ref,
                "curve_type": CurveType.REGULAR,
                "curve_style": CurvePlotStyle.STEP,
                "mpl_options": ref_options | {"where": "post"},
            },
        },
    )
    simulation_plot.add_subplot(
        row_idx=2,
        col_idx=2,
        subplot_name="delta",
        subplot_type=SubplotType.TEMPORAL,
        unit="°",
        show_unit=True,
        curves={
            r"$\delta$": {
                "data": np.rad2deg(delta),
                "curve_type": CurveType.REGULAR,
                "curve_style": CurvePlotStyle.STEP,
                "mpl_options": pred_options | {"where": "post"},
            },
        },
    )
    simulation_plot.add_subplot(
        row_idx=0,
        col_idx=2,
        subplot_name="r",
        subplot_type=SubplotType.TEMPORAL,
        unit="°/s",
        show_unit=True,
        curves={
            r"predicted $r$": {
                "data": np.rad2deg(x0toNf[:, 5]),
                "curve_type": CurveType.REGULAR,
                "curve_style": CurvePlotStyle.PLOT,
                "mpl_options": {"color": "red", "linewidth": 2},
            },
        },
    )
    simulation_plot.add_subplot(
        row_idx=1,
        col_idx=2,
        subplot_name="v_y",
        subplot_type=SubplotType.TEMPORAL,
        unit="m/s",
        show_unit=True,
        curves={
            r"predicted $v_y$": {
                "data": x0toNf[:, 4],
                "curve_type": CurveType.REGULAR,
                "curve_style": CurvePlotStyle.PLOT,
                "mpl_options": {"color": "red", "linewidth": 2},
            },
        },
    )
    simulation_plot.add_subplot(
        row_idx=3,
        col_idx=1,
        subplot_name="ddelta",
        subplot_type=SubplotType.TEMPORAL,
        unit="°/s",
        show_unit=True,
        curves={
            r"$d\delta$": {
                "data": np.rad2deg(ddelta),
                "curve_type": CurveType.REGULAR,
                "curve_style": CurvePlotStyle.STEP,
                "mpl_options": pred_options | {"where": "post"},
            },
            r"$d\delta_{ref}$": {
                "data": np.rad2deg(ddelta_ref),
                "curve_type": CurveType.REGULAR,
                "curve_style": CurvePlotStyle.STEP,
                "mpl_options": ref_options | {"where": "post"},
            },
        },
    )
    simulation_plot.plot(show=False)
