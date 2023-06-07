# Copyright (c) 2023 Tudor Oancea
import argparse
import json
import os
from copy import copy

import lightning as L
import torch
import torch.nn.functional as F
import wandb
from lightning import Fabric
from matplotlib import pyplot as plt
from tqdm import tqdm

from math_591_project.data_utils import *
from math_591_project.models import *
from math_591_project.plot_utils import *

L.seed_everything(127)


def run_model(system_model, batch):
    xtilde0, utilde0toNfminus1, xtilde1toNf = batch
    xtilde1toNf_p = system_model(xtilde0, utilde0toNfminus1)
    XY_loss = F.mse_loss(xtilde1toNf_p[..., :2], xtilde1toNf[..., :2])
    phi_loss = F.mse_loss(xtilde1toNf_p[..., 2], xtilde1toNf[..., 2])
    v_x_loss = F.mse_loss(xtilde1toNf_p[..., 3], xtilde1toNf[..., 3])
    if xtilde1toNf_p.shape[-1] > 4:
        v_y_loss = F.mse_loss(xtilde1toNf_p[..., 4], xtilde1toNf[..., 4])
        r_loss = F.mse_loss(xtilde1toNf_p[..., 5], xtilde1toNf[..., 5])
    else:
        v_y_loss = torch.tensor(0.0)
        r_loss = torch.tensor(0.0)
    return XY_loss, phi_loss, v_x_loss, v_y_loss, r_loss


def train(
    fabric: Fabric,
    model_name: str,
    system_model: OpenLoop,
    optimizer: torch.optim.Optimizer,
    train_dataloader: torch.utils.data.DataLoader,
    val_dataloader: torch.utils.data.DataLoader,
    scheduler: torch.optim.lr_scheduler.LRScheduler = None,
    num_epochs=10,
    loss_weights=[1.0, 1.0, 1.0, 1.0, 1.0],
    with_wandb=True,
):
    best_val_loss = np.inf
    for epoch in tqdm(range(num_epochs)):
        system_model.train()
        train_losses = {
            "total": 0.0,
            "XY": 0.0,
            "phi": 0.0,
            "v_x": 0.0,
            "v_y": 0.0,
            "r": 0.0,
        }
        for batch in train_dataloader:
            optimizer.zero_grad()
            XY_loss, phi_loss, v_x_loss, v_y_loss, r_loss = run_model(
                system_model, batch
            )
            loss = (
                loss_weights[0] * XY_loss
                + loss_weights[1] * phi_loss
                + loss_weights[2] * v_x_loss
                + loss_weights[3] * v_y_loss
                + loss_weights[4] * r_loss
            )
            fabric.backward(loss)
            optimizer.step()
            train_losses["total"] += loss.item()
            train_losses["XY"] += XY_loss.item()
            train_losses["phi"] += phi_loss.item()
            train_losses["v_x"] += v_x_loss.item()
            train_losses["v_y"] += v_y_loss.item()
            train_losses["r"] += r_loss.item()

        for k in train_losses:
            train_losses[k] /= len(train_dataloader)

        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.StepLR) or isinstance(
                scheduler, torch.optim.lr_scheduler.MultiStepLR
            ):
                scheduler.step()
            else:
                raise NotImplementedError("Scheduler not implemented")

        system_model.eval()
        val_losses = {
            "total": 0.0,
            "XY": 0.0,
            "phi": 0.0,
            "v_x": 0.0,
            "v_y": 0.0,
            "r": 0.0,
        }
        for batch in val_dataloader:
            with torch.no_grad():
                XY_loss, phi_loss, v_x_loss, v_y_loss, r_loss = run_model(
                    system_model, batch
                )
            val_losses["total"] += (
                loss_weights[0] * XY_loss
                + loss_weights[1] * phi_loss
                + loss_weights[2] * v_x_loss
                + loss_weights[3] * v_y_loss
                + loss_weights[4] * r_loss
            ).item()
            val_losses["XY"] += XY_loss.item()
            val_losses["phi"] += phi_loss.item()
            val_losses["v_x"] += v_x_loss.item()
            val_losses["v_y"] += v_y_loss.item()
            val_losses["r"] += r_loss.item()

        for k in val_losses:
            val_losses[k] /= len(val_dataloader)

        to_log = {}
        for k in train_losses:
            if k == "total":
                to_log[f"train_loss"] = train_losses[k]
                to_log[f"val_loss"] = val_losses[k]
            else:
                to_log[f"train/{k}_loss"] = train_losses[k]
                to_log[f"val/{k}_loss"] = val_losses[k]

        fabric.log_dict(to_log, step=epoch + 1)
        if with_wandb:
            to_log["epoch"] = epoch + 1
            if scheduler is not None:
                to_log["lr"] = copy(optimizer.param_groups[0]["lr"])
            wandb.log(to_log)

        if val_losses["total"] < best_val_loss:
            best_val_loss = val_losses["total"]
            fabric.save(
                f"checkpoints/{model_name}_best.ckpt",
                {
                    "system_model": system_model.model.ode,
                    "optimizer": optimizer,
                    "scheduler": scheduler,
                },
            )


def main():
    parser = argparse.ArgumentParser(description="arg parser")
    parser.add_argument(
        "--cfg_file",
        type=str,
        default="config/sysid_train_config.json",
        help="specify the config for training",
    )
    args = parser.parse_args()

    # set up parameters =========================================================
    config = json.load(open(args.cfg_file, "r"))
    with_wandb = config.pop("with_wandb")
    print(f"Training " + ("with" if with_wandb else "without") + " wandb")
    model_name: str = config["model"]["name"]
    if model_name.startswith("blackbox"):
        nhidden = config["model"]["n_hidden"]
        nonlinearity = config["model"]["nonlinearity"]
        from_checkpoint = config["model"]["from_checkpoint"]

    num_epochs = config["training"]["num_epochs"]
    loss_weights = config["training"]["loss_weights"]
    optimizer_params = config["training"]["optimizer"]
    optimizer = optimizer_params.pop("name")
    scheduler_params = config["training"]["scheduler"]
    scheduler = scheduler_params.pop("name")
    train_dataset_path = config["data"]["train"]
    test_dataset_path = config["data"]["test"]
    dt = 1 / 20

    # initialize wandb ==========================================
    if with_wandb:
        wandb.init(
            project="brains_neural_control",
            name=f"sysid|{model_name}",
            config=config,
        )

    # intialize lightning fabric ===============================================
    fabric = Fabric()
    print("Using device: ", fabric.device)

    # initialize model and optimizer =========================================================
    # match model_name:
    #     case "kin4":
    #         nxtilde = KIN4_NXTILDE
    #         nutilde = KIN4_NUTILDE
    #         ode_t = Kin4ODE
    #     case "dyn6":
    #         nxtilde = DYN6_NXTILDE
    #         nutilde = DYN6_NUTILDE
    #         ode_t = Dyn6ODE
    #     case "blackbox_kin4":
    #         nxtilde = KIN4_NXTILDE
    #         nutilde = KIN4_NUTILDE
    #         nin = nxtilde + nutilde
    #         nout = nxtilde
    #         ode_t = BlackboxKin4ODE
    #     case "blackbox_dyn6":
    #         nxtilde = DYN6_NXTILDE
    #         nutilde = DYN6_NUTILDE
    #         nin = nxtilde + nutilde - 3
    #         nout = nxtilde - 3
    #         ode_t = BlackboxDyn6ODE
    #     case _:
    #         raise ValueError(f"Unknown model name: {model_name}")
    dims = ode_dims[model_name]
    if model_name.startswith("blackbox"):
        ode_t, nxtilde, nutilde, nin, nout = dims
    else:
        ode_t, nxtilde, nutilde = dims

    system_model = OpenLoop(
        model=RK4(
            nxtilde=nxtilde,
            nutilde=nutilde,
            ode=ode_t(
                net=MLP(nin=nin, nout=nout, nhidden=nhidden, nonlinearity=nonlinearity)
            )
            if model_name.startswith("blackbox")
            else ode_t(),
            dt=dt,
        ),
        Nf=1,
    )
    if with_wandb:
        wandb.watch(system_model, log_freq=1)

    if model_name.startswith("blackbox") and from_checkpoint:
        try:
            system_model.model.ode.load_state_dict(
                torch.load(f"checkpoints/{model_name}_best.ckpt")["system_model"]
            )
            print("Successfully loaded model parameters from checkpoint")
        except FileNotFoundError:
            print("No checkpoint found, using random initialization")
        except RuntimeError:
            print("Checkpoint found, but not compatible with current model")

    match optimizer:
        case "sgd":
            optimizer_t = torch.optim.SGD
        case "adam":
            optimizer_t = torch.optim.Adam
        case "adamw":
            optimizer_t = torch.optim.AdamW
        case "lbfgs":
            optimizer_t = torch.optim.LBFGS
        case _:
            raise NotImplementedError(f"Optimizer {optimizer} not implemented")

    optimizer = optimizer_t(system_model.parameters(), **optimizer_params)

    match scheduler:
        case "steplr":
            scheduler_t = torch.optim.lr_scheduler.StepLR
        case "multisteplr":
            scheduler_t = torch.optim.lr_scheduler.MultiStepLR
        case _:
            scheduler_t = None
    scheduler = scheduler_t(optimizer, **scheduler_params) if scheduler_t else None

    system_model, optimizer = fabric.setup(system_model, optimizer)

    # load data ================================================================
    # load all CSV files in data/sysid
    data_dir = "data/" + train_dataset_path
    file_paths = [
        os.path.abspath(os.path.join(data_dir, file_path))
        for file_path in os.listdir(data_dir)
        if file_path.endswith(".csv")
    ]
    assert all([os.path.exists(path) for path in file_paths])
    train_dataloader, val_dataloader = get_sysid_loaders(file_paths)
    train_dataloader, val_dataloader = fabric.setup_dataloaders(
        train_dataloader, val_dataloader
    )

    # Run training loop with validation =========================================
    try:
        train(
            fabric,
            model_name,
            system_model,
            optimizer,
            train_dataloader,
            val_dataloader,
            scheduler=scheduler,
            num_epochs=num_epochs,
            loss_weights=loss_weights,
            with_wandb=with_wandb,
        )
    except KeyboardInterrupt:
        print(
            "Training interrupted by ctrl+C, saving model and proceeding to evaluation"
        )

    # save model ================================================================
    fabric.save(
        f"checkpoints/{model_name}_final.ckpt",
        {"system_model": system_model.model.ode},
    )

    if with_wandb:
        # log the model to wandb
        wandb.save(f"checkpoints/{model_name}_final.ckpt")
        wandb.save(f"checkpoints/{model_name}_best.ckpt")

    # evaluate model on test set ================================================
    # recreate open loop model with new Nf
    Nf = 40
    system_model = OpenLoop(
        model=RK4(
            nxtilde=nxtilde,
            nutilde=nutilde,
            ode=ode_t(
                net=MLP(nin=nin, nout=nout, nhidden=nhidden, nonlinearity=nonlinearity)
            )
            if model_name.startswith("blackbox")
            else ode_t(),
            dt=dt,
        ),
        Nf=Nf,
    )
    system_model.model.ode.load_state_dict(
        torch.load(f"checkpoints/{model_name}_best.ckpt")["system_model"]
    )
    system_model = fabric.setup(system_model)

    # create test dataloader
    data_dir = "data/" + test_dataset_path
    file_paths = [
        os.path.abspath(os.path.join(data_dir, file_path))
        for file_path in os.listdir(data_dir)
        if file_path.endswith(".csv")
    ]
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
    plot_names = [f"{model_name}_{i}.png" for i in range(5)]
    for i in range(5):
        (plot_kin4 if model_name.endswith("kin4") else plot_dyn6)(
            xtilde0=xtilde0[i],
            utilde0toNfminus1=utilde0toNfminus1[i],
            xtilde1toNf=xtilde1toNf[i],
            xtilde1toNf_p=xtilde1toNf_p[i],
            dt=dt,
        )
        plt.savefig("plots/" + plot_names[i], dpi=300)
    if with_wandb:
        # log the plot to wandb
        wandb.log(
            {
                "plot/" + plot_name: wandb.Image("plots/" + plot_name)
                for plot_name in plot_names
            }
        )

    plt.show()


if __name__ == "__main__":
    main()
