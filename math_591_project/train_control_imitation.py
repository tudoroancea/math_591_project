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


def run_model(model, batch):
    x0, xref0toNf, uref0toNfminus1 = batch
    u0toNfminus1 = model(x0, xref0toNf)
    T_loss = F.mse_loss(u0toNfminus1[:, :, 0], uref0toNfminus1[:, :, 0])
    ddelta_loss = F.mse_loss(u0toNfminus1[:, :, 1], uref0toNfminus1[:, :, 1])
    return T_loss, ddelta_loss


def train(
    fabric: Fabric,
    system_model_name: str,
    control_model: MLPControlPolicy,
    optimizer: torch.optim.Optimizer,
    train_dataloader: torch.utils.data.DataLoader,
    val_dataloader: torch.utils.data.DataLoader,
    scheduler: torch.optim.lr_scheduler.LRScheduler = None,
    num_epochs=10,
    loss_weights=(1.0, 1.0),
    with_wandb=False,
):
    best_val_loss = float("inf")
    for epoch in tqdm(range(num_epochs)):
        control_model.train()
        train_loss = 0.0
        train_T_loss = 0.0
        train_ddelta_loss = 0.0
        for batch in train_dataloader:
            optimizer.zero_grad()
            T_loss, ddelta_loss = run_model(control_model, batch)
            loss = loss_weights[0] * T_loss + loss_weights[1] * ddelta_loss
            fabric.backward(loss)
            optimizer.step()
            train_loss += loss.item()
            train_T_loss = T_loss.item()
            train_ddelta_loss = ddelta_loss.item()

        train_loss /= len(train_dataloader)
        train_T_loss /= len(train_dataloader)
        train_ddelta_loss /= len(train_dataloader)

        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.StepLR) or isinstance(
                scheduler, torch.optim.lr_scheduler.MultiStepLR
            ):
                scheduler.step()
            else:
                raise NotImplementedError("Scheduler not implemented")

        control_model.eval()
        val_T_loss = 0.0
        val_ddelta_loss = 0.0
        for batch in val_dataloader:
            with torch.no_grad():
                pose_loss, velocity_loss = run_model(control_model, batch)
            val_T_loss += pose_loss.item()
            val_ddelta_loss += velocity_loss.item()
        val_T_loss /= len(val_dataloader)
        val_ddelta_loss /= len(val_dataloader)
        val_loss = loss_weights[0] * val_T_loss + loss_weights[1] * val_ddelta_loss

        to_log = {
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train/T_loss": train_T_loss,
            "train/ddelta_loss": train_ddelta_loss,
            "val/T_loss": val_T_loss,
            "val/ddelta_loss": val_ddelta_loss,
        }
        fabric.log_dict(to_log, step=epoch + 1)
        if with_wandb:
            to_log["epoch"] = epoch + 1
            if scheduler is not None:
                to_log["lr"] = copy(optimizer.param_groups[0]["lr"])
            wandb.log(to_log)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            fabric.save(
                f"checkpoints/{system_model_name}_control_imitation_best.ckpt",
                {
                    "control_model": control_model.mlp,
                    "optimizer": optimizer,
                    "scheduler": scheduler,
                },
            )


def main():
    torch.autograd.set_detect_anomaly(True)

    parser = argparse.ArgumentParser(description="arg parser")
    parser.add_argument(
        "--cfg_file",
        type=str,
        default="config/control_imitation_train_config.json",
        help="specify the config for training",
    )
    args = parser.parse_args()

    # set up config =========================================================
    config = json.load(open(args.cfg_file, "r"))
    with_wandb = config["with_wandb"]
    Nf = 40
    dt = 1 / 20
    system_model_name = config["system_model"]["name"]
    control_model_best_path = (
        f"checkpoints/{system_model_name}_control_imitation_best.ckpt"
    )
    control_model_final_path = (
        f"checkpoints/{system_model_name}_control_imitation_final.ckpt"
    )
    nhidden = config["control_model"]["nhidden"]
    nonlinearity = config["control_model"]["nonlinearity"]
    from_checkpoint = config["control_model"]["from_checkpoint"]
    num_epochs = config["training"]["num_epochs"]
    optimizer_params = config["training"]["optimizer"]
    optimizer = optimizer_params.pop("name")
    scheduler_params = config["training"]["scheduler"]
    scheduler = scheduler_params.pop("name")
    train_dataset_path = config["data"]["train"]
    test_dataset_path = config["data"]["test"]

    # initialize wandb ==========================================
    if with_wandb:
        wandb.init(
            project="brains_neural_control",
            name=f"control|imitation",
            config=config,
        )

    # intialize lightning fabric ===============================================
    fabric = Fabric()

    # initialize model and optimizer =========================================================
    match system_model_name:
        case "kin4":
            nxtilde = KIN4_NXTILDE
            nutilde = KIN4_NUTILDE
            ode_t = Kin4ODE
        case "dyn6":
            nxtilde = DYN6_NXTILDE
            nutilde = DYN6_NUTILDE
            ode_t = Dyn6ODE
        case "blackbox_kin4":
            nxtilde = KIN4_NXTILDE
            nutilde = KIN4_NUTILDE
            nin = nxtilde + nutilde
            nout = nxtilde
            ode_t = BlackboxKin4ODE
        case "blackbox_dyn6":
            nxtilde = DYN6_NXTILDE
            nutilde = DYN6_NUTILDE
            nin = nxtilde + nutilde - 3
            nout = nxtilde - 3
            ode_t = BlackboxDyn6ODE
        case _:
            raise ValueError(f"Unknown model name: {system_model_name}")
    nx = nxtilde + 1
    nu = nutilde

    control_mlp = MLP(
        nin=nx + (Nf + 1) * 4,
        nout=nu * Nf,
        nhidden=nhidden,
        nonlinearity=nonlinearity,
    )
    if from_checkpoint and os.path.exists(control_model_best_path):
        control_mlp.load_state_dict(
            torch.load(control_model_best_path)["control_model"]
        )
    control_model = MLPControlPolicy(nx=nx, nu=nu, Nf=Nf, mlp=control_mlp)
    if with_wandb:
        wandb.watch(control_model, log_freq=1)

    match optimizer:
        case "sgd":
            optimizer_type = torch.optim.SGD
        case "adam":
            optimizer_type = torch.optim.Adam
        case "adamw":
            optimizer_type = torch.optim.AdamW
        case _:
            raise NotImplementedError(f"Optimizer {optimizer} not implemented")

    optimizer = optimizer_type(control_model.parameters(), **optimizer_params)

    match scheduler:
        case "steplr":
            scheduler_type = torch.optim.lr_scheduler.StepLR
        case "multisteplr":
            scheduler_type = torch.optim.lr_scheduler.MultiStepLR
        case _:
            scheduler_type = None
    scheduler = (
        scheduler_type(optimizer, **scheduler_params) if scheduler_type else None
    )

    control_model, optimizer = fabric.setup(control_model, optimizer)

    # load data ================================================================
    # load all CSV files in data/sysid
    data_dir = "data/" + train_dataset_path
    file_paths = [
        os.path.abspath(os.path.join(data_dir, file_path))
        for file_path in os.listdir(data_dir)
        if file_path.endswith(".csv")
    ]
    assert all([os.path.exists(path) for path in file_paths])
    train_dataloader, val_dataloader = get_control_loaders(file_paths)
    train_dataloader, val_dataloader = fabric.setup_dataloaders(
        train_dataloader, val_dataloader
    )

    # Run training loop with validation =========================================
    train(
        fabric,
        system_model_name,
        control_model,
        optimizer,
        train_dataloader,
        val_dataloader,
        scheduler=scheduler,
        num_epochs=num_epochs,
        loss_weights=(1.0, 1.0),
        with_wandb=with_wandb,
    )

    # save model ================================================================
    fabric.save(
        control_model_final_path,
        {"control_model": control_model.mlp},
    )

    if with_wandb:
        # log the model to wandb
        wandb.save(control_model_best_path)
        wandb.save(control_model_final_path)

    # evaluate model on test set ================================================
    # load best control model
    control_model.mlp.load_state_dict(
        torch.load(control_model_best_path)["control_model"]
    )
    control_model.eval()
    # load system model
    system_model = OpenLoop(
        model=ControlDiscreteModel(
            nx=nx,
            nu=nu,
            model=RK4(
                nxtilde=nxtilde,
                nutilde=nutilde,
                ode=ode_t(
                    net=MLP(
                        nin=nin,
                        nout=nout,
                        nhidden=config["system_model"]["nhidden"],
                        nonlinearity=config["system_model"]["nonlinearity"],
                    ),
                )
                if system_model_name.startswith("blackbox")
                else ode_t(),
                dt=dt,
            ),
        ),
        Nf=Nf,
    )
    system_model.model.model.ode.load_state_dict(
        torch.load(f"checkpoints/{system_model_name}_best.ckpt")["system_model"]
    )
    # for p in system_model.parameters():
    #     p.requires_grad_(False)
    system_model.requires_grad_(False)
    system_model.eval()
    system_model = fabric.setup(system_model)

    # load test train_dataset
    data_dir = "data/" + test_dataset_path
    file_paths = [
        os.path.abspath(os.path.join(data_dir, file_path))
        for file_path in os.listdir(data_dir)
        if file_path.endswith(".csv")
    ]
    test_dataset = ControlDataset(file_paths)
    test_dataloader = DataLoader(
        test_dataset, batch_size=5, shuffle=True, num_workers=0
    )
    test_dataloader = fabric.setup_dataloaders(test_dataloader)

    # evaluate model on test set
    x0, xref0toNf, uref0toNfminus1 = next(iter(test_dataloader))
    with torch.no_grad():
        u0toNfminus1 = control_model(x0, xref0toNf)
    x1toNf = system_model(x0, u0toNfminus1)

    x0 = x0.detach().cpu().numpy()
    xref0toNf = xref0toNf.detach().cpu().numpy()
    u0toNfminus1 = u0toNfminus1.detach().cpu().numpy()
    uref0toNfminus1 = uref0toNfminus1.detach().cpu().numpy()
    x1toNf = x1toNf.detach().cpu().numpy()
    if not os.path.exists("plots"):
        os.mkdir("plots")
    plot_names = [f"{system_model_name}_control_imitation_{i}.png" for i in range(5)]
    for i in range(5):
        (plot_kin4_control if system_model_name == "kin4" else plot_dyn6_control)(
            x0=x0[i],
            xref0toNf=xref0toNf[i],
            u0toNfminus1=u0toNfminus1[i],
            x1toNf=x1toNf[i],
            uref0toNfminus1=uref0toNfminus1[i],
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
