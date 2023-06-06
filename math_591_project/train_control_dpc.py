# Copyright (c) 2023 Tudor Oancea
import argparse
import json
import os
from copy import copy

import lightning as L
import torch
import torch.nn.functional as F
import wandb
from icecream import ic
from lightning import Fabric
from matplotlib import pyplot as plt
from tqdm import tqdm

from math_591_project.data_utils import *
from math_591_project.models import *
from math_591_project.plot_utils import *

L.seed_everything(127)


def run_model(system_model, control_model, batch, delta_max, ddelta_max):
    x0, xref0toNf, _ = batch
    u0toNfminus1 = control_model(x0, xref0toNf)
    x1toNf = system_model(x0, u0toNfminus1)
    # reference losses
    XY_loss = F.mse_loss(x1toNf[:, :, :2], xref0toNf[:, 1:, :2])
    phi_loss = F.mse_loss(x1toNf[:, :, 2], xref0toNf[:, 1:, 2])
    v_x_loss = F.mse_loss(x1toNf[:, :, 3], xref0toNf[:, 1:, 3])
    # control inputs penalties
    delta_loss = torch.mean(x1toNf[:, :, -1] ** 2)
    T_loss = torch.mean(u0toNfminus1[:, :, 0] ** 2)
    ddelta_loss = torch.mean(u0toNfminus1[:, :, 1] ** 2)
    # constraints violation penalties
    v_x_ub_loss = torch.mean(F.relu(x1toNf[:, :, 3] - 15.0) ** 2)
    v_x_lb_loss = torch.mean(F.relu(-x1toNf[:, :, 3]) ** 2)
    delta_ub_loss = torch.mean(F.relu(x1toNf[:, :, -1] - delta_max) ** 2)
    delta_lb_loss = torch.mean(F.relu(-x1toNf[:, :, -1] - delta_max) ** 2)
    ddelta_ub_loss = torch.mean(F.relu(u0toNfminus1[:, :, 1] - ddelta_max) ** 2)
    ddelta_lb_loss = torch.mean(F.relu(-u0toNfminus1[:, :, 1] - ddelta_max) ** 2)

    return (
        XY_loss,
        phi_loss,
        v_x_loss,
        delta_loss,
        T_loss,
        ddelta_loss,
        v_x_ub_loss,
        v_x_lb_loss,
        delta_ub_loss,
        delta_lb_loss,
        ddelta_ub_loss,
        ddelta_lb_loss,
    )


def train(
    fabric: Fabric,
    system_model_name: str,
    system_model: torch.nn.Module,
    control_model: MLPControlPolicy,
    optimizer: torch.optim.Optimizer,
    train_dataloader: torch.utils.data.DataLoader,
    val_dataloader: torch.utils.data.DataLoader,
    loss_weights: dict,
    scheduler: torch.optim.lr_scheduler.LRScheduler = None,
    num_epochs: int = 10,
    with_wandb: bool = False,
):
    ic(loss_weights)
    delta_max = torch.deg2rad(
        torch.tensor(
            40.0, dtype=torch.float32, requires_grad=False, device=control_model.device
        )
    )
    ddelta_max = (
        torch.deg2rad(
            torch.tensor(
                68.0,
                dtype=torch.float32,
                requires_grad=False,
                device=control_model.device,
            )
        )
        / 20.0
    )
    best_val_loss = float("inf")
    for epoch in tqdm(range(num_epochs)):
        control_model.train()
        train_losses = {
            "XY": 0.0,
            "phi": 0.0,
            "v_x": 0.0,
            "delta": 0.0,
            "T": 0.0,
            "ddelta": 0.0,
            "v_x_ub": 0.0,
            "v_x_lb": 0.0,
            "delta_ub": 0.0,
            "delta_lb": 0.0,
            "ddelta_ub": 0.0,
            "ddelta_lb": 0.0,
            "total": 0.0,
        }
        val_losses = copy(train_losses)
        for batch in train_dataloader:
            optimizer.zero_grad()
            (
                XY_loss,
                phi_loss,
                v_x_loss,
                delta_loss,
                T_loss,
                ddelta_loss,
                v_x_ub_loss,
                v_x_lb_loss,
                delta_ub_loss,
                delta_lb_loss,
                ddelta_ub_loss,
                ddelta_lb_loss,
            ) = run_model(system_model, control_model, batch, delta_max, ddelta_max)

            loss = (
                loss_weights["q_XY"] * XY_loss
                + loss_weights["q_phi"] * phi_loss
                + loss_weights["q_v_x"] * v_x_loss
                + loss_weights["q_delta"] * delta_loss
                + loss_weights["q_T"] * T_loss
                + loss_weights["q_ddelta"] * ddelta_loss
                + loss_weights["q_s"]
                * (
                    v_x_lb_loss
                    + v_x_ub_loss
                    + delta_lb_loss
                    + delta_ub_loss
                    + ddelta_lb_loss
                    + ddelta_ub_loss
                )
            )
            fabric.backward(loss)
            optimizer.step()
            train_losses["XY"] += XY_loss.item()
            train_losses["phi"] += phi_loss.item()
            train_losses["v_x"] += v_x_loss.item()
            train_losses["delta"] += delta_loss.item()
            train_losses["T"] += T_loss.item()
            train_losses["ddelta"] += ddelta_loss.item()
            train_losses["v_x_ub"] += v_x_ub_loss.item()
            train_losses["v_x_lb"] += v_x_lb_loss.item()
            train_losses["delta_ub"] += delta_ub_loss.item()
            train_losses["delta_lb"] += delta_lb_loss.item()
            train_losses["ddelta_ub"] += ddelta_ub_loss.item()
            train_losses["ddelta_lb"] += ddelta_lb_loss.item()
            train_losses["total"] += loss.item()

        for k in train_losses.keys():
            train_losses[k] /= len(train_dataloader)

        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.StepLR) or isinstance(
                scheduler, torch.optim.lr_scheduler.MultiStepLR
            ):
                scheduler.step()
            else:
                raise NotImplementedError("Scheduler not implemented")

        control_model.eval()
        for batch in val_dataloader:
            with torch.no_grad():
                (
                    XY_loss,
                    phi_loss,
                    v_x_loss,
                    delta_loss,
                    T_loss,
                    ddelta_loss,
                    v_x_ub_loss,
                    v_x_lb_loss,
                    delta_ub_loss,
                    delta_lb_loss,
                    ddelta_ub_loss,
                    ddelta_lb_loss,
                ) = run_model(system_model, control_model, batch, delta_max, ddelta_max)
            val_losses["XY"] += XY_loss.item()
            val_losses["phi"] += phi_loss.item()
            val_losses["v_x"] += v_x_loss.item()
            val_losses["delta"] += delta_loss.item()
            val_losses["T"] += T_loss.item()
            val_losses["ddelta"] += ddelta_loss.item()
            val_losses["v_x_ub"] += v_x_ub_loss.item()
            val_losses["v_x_lb"] += v_x_lb_loss.item()
            val_losses["delta_ub"] += delta_ub_loss.item()
            val_losses["delta_lb"] += delta_lb_loss.item()
            val_losses["ddelta_ub"] += ddelta_ub_loss.item()
            val_losses["ddelta_lb"] += ddelta_lb_loss.item()
            val_losses["total"] += (
                loss_weights["q_XY"] * XY_loss
                + loss_weights["q_phi"] * phi_loss
                + loss_weights["q_v_x"] * v_x_loss
                + loss_weights["q_delta"] * delta_loss
                + loss_weights["q_T"] * T_loss
                + loss_weights["q_ddelta"] * ddelta_loss
                + loss_weights["q_s"]
                * (v_x_lb_loss + v_x_ub_loss + delta_lb_loss + delta_ub_loss)
            ).item()

        for k in val_losses.keys():
            val_losses[k] /= len(val_dataloader)

        to_log = {}
        for k in train_losses.keys():
            if k == "total":
                to_log["train_loss"] = train_losses[k]
                to_log["val_loss"] = val_losses[k]
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
                f"checkpoints/{system_model_name}_control_best.ckpt",
                {
                    "control_model": control_model.mlp,
                    "optimizer": optimizer,
                    "scheduler": scheduler,
                },
            )


def main():
    parser = argparse.ArgumentParser(description="arg parser")
    parser.add_argument(
        "--cfg_file",
        type=str,
        default="config/control_dpc_train_config.json",
        help="specify the config for training",
    )
    parser.add_argument(
        "--control_ckpt",
        type=str,
        default="",
    )
    args = parser.parse_args()

    # set up config =========================================================
    config = json.load(open(args.cfg_file, "r"))
    with_wandb = config.pop("with_wandb")
    dt = 1 / 20
    Nf = 40
    system_model_name = config["system_model"]["name"]
    control_model_best_path = f"checkpoints/{system_model_name}_control_best.ckpt"
    control_model_final_path = f"checkpoints/{system_model_name}_control_final.ckpt"
    num_epochs = config["training"]["num_epochs"]
    optimizer_params = config["training"]["optimizer"]
    optimizer = optimizer_params.pop("name")
    scheduler_params = config["training"]["scheduler"]
    scheduler = scheduler_params.pop("name")
    train_dataset_path = config["data"]["train"]
    test_dataset_path = config["data"]["test"]
    loss_weights = config["training"]["loss_weights"]

    # initialize wandb ==========================================
    if with_wandb:
        wandb.init(
            project="brains_neural_control",
            name=f"control|dpc",
            config=config,
        )

    # intialize lightning fabric ===============================================
    fabric = Fabric()
    print("Fabric initialized with devices:", fabric.device)

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
    #     p.requires_grad = False
    system_model.requires_grad_(False)
    system_model.eval()
    control_mlp = MLP(
        nin=nx - 3 + (Nf + 1) * 4,
        nout=nu * Nf,
        nhidden=config["control_model"]["nhidden"],
        nonlinearity=config["control_model"]["nonlinearity"],
    )
    if config["control_model"]["from_checkpoint"]:
        if args.control_ckpt != "":
            path = args.control_ckpt
        else:
            path = control_model_best_path
        if os.path.exists(path):
            control_mlp.load_state_dict(torch.load(path)["control_model"])
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
    system_model = fabric.setup(system_model)

    # load data ================================================================
    # load all CSV files in data/sysid
    data_dir = "data/" + train_dataset_path
    file_paths = [
        os.path.abspath(os.path.join(data_dir, file_path))
        for file_path in os.listdir(data_dir)
        if file_path.endswith(".csv")
    ]
    assert all([os.path.exists(path) for path in file_paths])
    train_dataloader, val_dataloader = get_control_loaders(
        file_paths, batch_sizes=(10000, 5000)
    )
    train_dataloader, val_dataloader = fabric.setup_dataloaders(
        train_dataloader, val_dataloader
    )

    # Run training loop with validation =========================================
    train(
        fabric,
        system_model_name,
        system_model,
        control_model,
        optimizer,
        train_dataloader,
        val_dataloader,
        scheduler=scheduler,
        num_epochs=num_epochs,
        loss_weights=loss_weights,
        with_wandb=with_wandb,
    )
    # compare one weight in system model's state dict before and after training
    # to make sure it's not changing
    wbefore = torch.load(f"checkpoints/{system_model_name}_best.ckpt")["system_model"][
        "net.net.output_layer.weight"
    ]
    wafter = system_model.model.model.ode.net.net[-1].weight
    print(
        f"sytem model output layer weight is the same: {torch.allclose(wbefore, wafter)}"
    )
    print(f"grad is None: {wafter.grad is None}")

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
    # load best model
    control_model.mlp.load_state_dict(
        torch.load(control_model_best_path)["control_model"]
    )
    control_model.eval()
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
    x0, xref0toNf, _ = next(iter(test_dataloader))
    with torch.no_grad():
        u0toNfminus1 = control_model(x0, xref0toNf)
    x1toNf = system_model(x0, u0toNfminus1)

    x0 = x0.detach().cpu().numpy()
    xref0toNf = xref0toNf.detach().cpu().numpy()
    u0toNfminus1 = u0toNfminus1.detach().cpu().numpy()
    x1toNf = x1toNf.detach().cpu().numpy()
    if not os.path.exists("plots"):
        os.mkdir("plots")
    plot_names = [f"{system_model_name}_control_{i}.png" for i in range(5)]
    for i in range(5):
        (plot_kin4_control if system_model_name == "kin4" else plot_dyn6_control)(
            x0=x0[i],
            xref0toNf=xref0toNf[i],
            u0toNfminus1=u0toNfminus1[i],
            x1toNf=x1toNf[i],
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
