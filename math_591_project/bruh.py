import torch
from math_591_project.data_utils import *
import os
import pandas as pd
from icecream import ic

def main():
    # load the run data and find the first row with method == "DPC"
    run_data = pd.read_csv("bruh.csv") 
    id = run_data[run_data["method"] == "DPC"].index[0]
    # create from this row the variables x0 (columns X,Y,phi,v_x,v_y,r,last_delta) and xref0toNf (columns X_ref_0,Y_ref_0,phi_ref_0,v_x_ref_0,...,X_ref_Nf,Y_ref_Nf,phi_ref_Nf,v_x_ref_Nf)
    x0 = run_data[["X", "Y", "phi", "v_x", "v_y", "r", "last_delta"]].iloc[id].to_numpy()
    xref0toNf = run_data[[col for col in run_data.columns if "ref" in col]].iloc[id].to_numpy()
    x0 = torch.from_numpy(x0).to(dtype=torch.float32)
    xref0toNf = torch.from_numpy(xref0toNf).to(dtype=torch.float32)
    bruh = torch.cat((x0, xref0toNf), dim=0).view(1, -1)

    # load training data
    prefix = "data_v1.1.0/train"
    file_paths = [
        os.path.join(prefix, f) for f in os.listdir(prefix) if f.endswith(".csv")
    ]
    dataset = ControlDataset(file_paths)
    bruh_ref = torch.cat((dataset.x0.squeeze(1), dataset.xref0toNf.view(dataset.x0.shape[0],-1)), dim=1)
    print(bruh.shape, bruh_ref.shape)

    dist = torch.norm(bruh - bruh_ref, dim=1)
    min_dist, min_idx = torch.min(dist, dim=0)
    min_dist = min_dist.item()
    min_idx = min_idx.item()
    print(min_dist, min_idx)
    print(f"x0: {bruh[0, :7]}\nx0_dataset: {bruh_ref[min_idx, :7]}\nxref0toNf: {bruh[0, 7:]}\nxref0toNf_dataset: {bruh_ref[min_idx, 7:]}")

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

if __name__ == "__main__":
    main()
