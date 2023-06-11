# Copyright (c) 2023 Tudor Oancea
import argparse
import json
import os

import torch
from lightning import Fabric
from matplotlib import pyplot as plt

from math_591_project.data_utils import *
from math_591_project.models import *
from math_591_project.plot_utils import *


def main():
    parser = argparse.ArgumentParser(description="arg parser")
    parser.add_argument(
        "--cfg_file",
        type=str,
        default="config/blackbox_dyn6_sysid.json",
        help="specify the config for testing",
    )
    args = parser.parse_args()

    # set up parameters =========================================================
    config = json.load(open(args.cfg_file, "r"))
    model_name: str = config["model"]["name"]
    if model_name.startswith("blackbox"):
        n_hidden = config["model"]["n_hidden"]
        nonlinearity = config["model"]["nonlinearity"]

    test_dataset_path = config["data"]["test"]
    testing_params = config["testing"]
    num_samples = testing_params["num_samples"]
    dt = 1 / 20
    test_Nf = testing_params["Nf"]

    dims = ode_dims[model_name]
    if model_name.startswith("blackbox"):
        ode_t, nxtilde, nutilde, nin, nout = dims
    else:
        ode_t, nxtilde, nutilde = dims

    fabric = Fabric()
    print(f"Using {fabric.device} device")

    system_model = OpenLoop(
        model=RK4(
            nxtilde=nxtilde,
            nutilde=nutilde,
            ode=ode_t(
                net=MLP(nin=nin, nout=nout, nhidden=n_hidden, nonlinearity=nonlinearity)
            )
            if model_name.startswith("blackbox")
            else ode_t(),
            dt=dt,
        ),
        Nf=test_Nf,
    )
    system_model.model.ode.load_state_dict(
        torch.load(f"checkpoints/{model_name}_best.ckpt", map_location="cpu")[
            "system_model"
        ]
    )
    system_model = fabric.setup(system_model)

    # create test dataloader
    data_dir = "data/" + test_dataset_path
    file_paths = [
        os.path.abspath(os.path.join(data_dir, file_path))
        for file_path in os.listdir(data_dir)
        if file_path.endswith(".csv")
    ]
    test_dataset = SysidTestDataset(file_paths, test_Nf)
    test_dataloader = DataLoader(
        test_dataset, batch_size=num_samples, shuffle=True, num_workers=1
    )
    test_dataloader = fabric.setup_dataloaders(test_dataloader)

    # evaluate model on test set
    system_model.eval()
    xtilde0, utilde0toNfminus1, xtilde1toNf = next(iter(test_dataloader))
    xtilde1toNf_p = system_model(xtilde0, utilde0toNfminus1)
    xtilde0 = xtilde0.detach().cpu().numpy()
    utilde0toNfminus1 = utilde0toNfminus1.detach().cpu().numpy()
    xtilde1toNf = xtilde1toNf.detach().cpu().numpy()
    xtilde1toNf_p = xtilde1toNf_p.detach().cpu().numpy()
    if not os.path.exists("test_plots"):
        os.mkdir("test_plots")
    plot_names = [f"test_plots/{model_name}_{i}.png" for i in range(num_samples)]
    for i in range(num_samples):
        (plot_kin4 if model_name.endswith("kin4") else plot_dyn6)(
            xtilde0=xtilde0[i],
            utilde0toNfminus1=utilde0toNfminus1[i],
            xtilde1toNf=xtilde1toNf[i],
            xtilde1toNf_p=xtilde1toNf_p[i],
            dt=dt,
        )
        plt.savefig(plot_names[i], dpi=300)

    # plt.show()


if __name__ == "__main__":
    main()
