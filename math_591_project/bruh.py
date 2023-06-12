from matplotlib.pyplot import pink
import torch
from math_591_project.data_utils import *
import os
import pandas as pd
from icecream import ic


def main():
    # load the run data and find the first row with method == "DPC"
    print("loading run data...")
    run_data = pd.read_csv("bruh3.csv")
    id = run_data[run_data["method"] == "DPC"].index[0]
    # create from this row the variables x0 (columns X,Y,phi,v_x,v_y,r,last_delta) and xref0toNf (columns X_ref_0,Y_ref_0,phi_ref_0,v_x_ref_0,...,X_ref_Nf,Y_ref_Nf,phi_ref_Nf,v_x_ref_Nf)
    x0 = (
        run_data[["X", "Y", "phi", "v_x", "v_y", "r", "last_delta"]].iloc[id].to_numpy()
    )
    xref0toNf = (
        run_data[[col for col in run_data.columns if "ref" in col]].iloc[id].to_numpy()
    )
    x0 = torch.from_numpy(x0).to(dtype=torch.float32)
    xref0toNf = torch.from_numpy(xref0toNf).to(dtype=torch.float32)
    xref0toNf = xref0toNf.view(-1,4)
    x0[2] = wrapToPi(x0[2])
    xref0toNf[:, 2] = wrapToPi(xref0toNf[:, 2])
    xref0toNf[:, 2] = teds_projection(xref0toNf[:, 2], x0[2] - np.pi)
    xref0toNf[:, 2] = unwrapToPi(xref0toNf[:, 2])
    bruh = torch.cat((x0, xref0toNf.view(-1)), dim=0).view(1, -1)

    # load training data
    print("loading training data...")
    prefix = "data_v1.1.0/train"
    file_paths = [
        os.path.join(prefix, f) for f in os.listdir(prefix) if f.endswith(".csv")
    ]
    dataset = ControlDataset(file_paths)
    bruh_ref = torch.cat(
        (dataset.x0.squeeze(1), dataset.xref0toNf.view(dataset.x0.shape[0], -1)), dim=1
    )

    # find the closest point in the training data to the run data
    print("finding closest point in training data...")
    dist = torch.norm(bruh - bruh_ref, dim=1)
    min_dist, min_idx = torch.min(dist, dim=0)
    min_dist = min_dist.item()
    min_idx = min_idx.item()
    print(f"min_dist: {min_dist}, min_idx: {min_idx}")
    print(
        f"x0: {bruh[0, :7]}\nx0_dataset: {bruh_ref[min_idx, :7]}\nxref0toNf: {bruh[0, 7:]}\nxref0toNf_dataset: {bruh_ref[min_idx, 7:]}"
    )

    # x0_dist = torch.norm(dataset.x0.squeeze(1) - x0, dim=1)
    # min_x0_dist, min_x0_idx = torch.min(x0_dist, dim=0)
    # min_x0_dist = min_x0_dist.item()
    # min_x0_idx = min_x0_idx.item()
    # print(min_x0_dist, min_x0_idx)
    # xref0toNf_dist = torch.norm(dataset.xref0toNf.view(dataset.xref0toNf.shape[0], -1) - xref0toNf, dim=1)
    # min_xref0toNf_dist, min_xref0toNf_idx = torch.min(xref0toNf_dist, dim=0)
    # min_xref0toNf_dist = min_xref0toNf_dist.item()
    # min_xref0toNf_idx = min_xref0toNf_idx.item()
    # print(min_xref0toNf_dist, min_xref0toNf_idx)


def main2():
    file_paths = [
        os.path.join("data_v1.1.0/train", f)
        for f in os.listdir("data_v1.1.0/train")
        if f.endswith(".csv")
    ]

    for file_path in file_paths:
        _, x, xref, uref = load_data(file_path)
        # check if there are values of phi or phi_ref that are not in (-pi, pi]
        in_range = np.logical_and(x[:, 2] > -np.pi, x[:, 2] <= np.pi)
        assert np.all(in_range), f"phi not good, wrong idx: {np.argwhere(~in_range)}"
        in_range = np.logical_and(xref[:, 2] > -np.pi, xref[:, 2] <= np.pi)
        assert np.all(
            in_range
        ), f"phi_ref not good, wrong idx: {np.argwhere(~in_range)} and values: {xref[~in_range, 2]}"


if __name__ == "__main__":
    main()
    # main2()
