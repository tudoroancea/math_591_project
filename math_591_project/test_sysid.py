# Copyright (c) 2023 Tudor Oancea


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from matplotlib import pyplot as plt
from math_591_project.data_utils import *
from math_591_project.models import *
from math_591_project.plot_utils import *

def test():
    # evaluate model on test set ================================================
    # recreate open loop model with new Nf
    Nf = 40
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
        Nf=Nf,
    )
    system_model.load_state_dict(
        torch.load(f"checkpoints/{model_name}_best.ckpt")["system_model"]
    )
    system_model = fabric.setup(system_model)

    # create test dataloader
    test_dataset = SysidTestDataset(file_paths, Nf)
    test_dataloader = DataLoader(
        test_dataset, batch_size=5, shuffle=True, num_workers=1
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
    if not os.path.exists("plots"):
        os.mkdir("plots")
    plot_names = [f"plots/{model_name}_{i}.png" for i in range(5)]
    for i in range(5):
        (plot_kin4 if model_name.endswith("kin4") else plot_dyn6)(
            xtilde0=xtilde0[i],
            utilde0toNfminus1=utilde0toNfminus1[i],
            xtilde1toNf=xtilde1toNf[i],
            xtilde1toNf_p=xtilde1toNf_p[i],
            dt=dt,
        )
        plt.savefig(plot_names[i], dpi=300)
    if with_wandb:
        # log the plot to wandb
        wandb.log(
            {"plot/" + plot_name: wandb.Image(plot_name) for plot_name in plot_names}
        )

    plt.show()
