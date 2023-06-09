# Copyright (c) 2023 Tudor Oancea
import argparse
import json
import os
from copy import copy
from pyexpat import model

import lightning as L
import torch
import torch.nn.functional as F
import wandb
from icecream import ic
from lightning import Fabric
from matplotlib import pyplot as plt
from tqdm import tqdm

from math_591_project.models import *
from math_591_project.utils import *

L.seed_everything(127)


def run_model(system_model, control_model, batch, delta_max):
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
    )


def train(
    fabric: Fabric,
    system_model: torch.nn.Module,
    control_model: MLPControlPolicy,
    optimizer: torch.optim.Optimizer,
    train_dataloader: torch.utils.data.DataLoader,
    val_dataloader: torch.utils.data.DataLoader,
    loss_weights: dict,
    control_output_ckpt: str,
    scheduler: torch.optim.lr_scheduler.LRScheduler = None,
    num_epochs: int = 10,
    with_wandb: bool = False,
):
    delta_max = torch.deg2rad(
        torch.tensor(
            40.0, dtype=torch.float32, requires_grad=False, device=control_model.device
        )
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
            ) = run_model(system_model, control_model, batch, delta_max)

            loss = (
                loss_weights["q_XY"] * XY_loss
                + loss_weights["q_phi"] * phi_loss
                + loss_weights["q_v_x"] * v_x_loss
                + loss_weights["q_delta"] * delta_loss
                + loss_weights["q_T"] * T_loss
                + loss_weights["q_ddelta"] * ddelta_loss
                + loss_weights["q_s"]
                * (v_x_lb_loss + v_x_ub_loss + delta_lb_loss + delta_ub_loss)
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
                ) = run_model(system_model, control_model, batch, delta_max)
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
                control_output_ckpt,
                control_model.mlp.state_dict(),
            )


def main():
    parser = argparse.ArgumentParser(description="arg parser")
    parser.add_argument(
        "--cfg_file",
        type=str,
        default="config/control/neural_dyn6.json",
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
    print("Training " + ("with" if with_wandb else "without") + " wandb")
    Nf = 40

    # system model config
    system_model_config = config["system_model"]
    system_model_name = system_model_config["name"]
    ode_t = ode_from_string[system_model_name]
    assert issubclass(ode_t, NeuralMixin), "system model must be a neural model"
    system_input_checkpoint = system_model_config["input_checkpoint"]

    # control model config
    control_model_config = config["control_model"]
    control_input_ckpt = control_model_config["input_checkpoint"]
    control_output_ckpt = control_model_config["output_checkpoint"]

    # training config
    train_config = config["training"]
    num_epochs = train_config["num_epochs"]
    loss_weights = train_config["loss_weights"]
    train_val_batch_size = train_config["batch_size"]
    train_data_dir = train_config["data_dir"]

    optimizer_params = train_config["optimizer"]
    optimizer = optimizer_params.pop("name")

    scheduler_params = train_config["scheduler"]
    scheduler = scheduler_params.pop("name")

    # testing config
    test_data_dir = config["testing"]["data_dir"]

    # intialize lightning fabric ===============================================
    fabric = Fabric()
    print("Fabric initialized with devices:", fabric.device)

    # initialize model and optimizer =========================================================
    system_model = OpenLoop(
        model=LiftedDiscreteModel(
            model=RK4(
                ode=ode_t(
                    net=MLP(
                        nin=ode_t.nin,
                        nout=ode_t.nout,
                        nhidden=system_model_config["nhidden"],
                        nonlinearity=system_model_config["nonlinearity"],
                    ),
                ),
                dt=dt,
            ),
        ),
        Nf=Nf,
    )
    try:
        system_model.model.model.ode.net.load_state_dict(
            torch.load(system_input_checkpoint, map_location="cpu")
        )
        print("Successfully loaded system model parameters from checkpoint")
    except FileNotFoundError:
        print("No checkpoint found for system model, using random initialization")
    except RuntimeError:
        print(
            "Checkpoint found for system model, but not compatible with current model"
        )

    system_model.requires_grad_(False)
    system_model.eval()

    control_model = MLPControlPolicy(
        mlp=MLP(
            nin=MLPControlPolicy.nin,
            nout=MLPControlPolicy.nout,
            nhidden=control_model_config["nhidden"],
            nonlinearity=control_model_config["nonlinearity"],
        ),
    )
    if control_model_config["from_checkpoint"]:
        try:
            control_model.mlp.load_state_dict(
                torch.load(
                    args.control_ckpt
                    if args.control_ckpt != ""
                    else control_input_ckpt,
                    map_location="cpu",
                )
            )
            print("Successfully loaded control model parameters from checkpoint")
        except FileNotFoundError:
            print("No checkpoint found for control model, using random initialization")
        except RuntimeError:
            print(
                "Checkpoint found for control model, but not compatible with current model"
            )

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
    file_paths = [
        os.path.abspath(os.path.join(train_data_dir, file_path))
        for file_path in os.listdir(train_data_dir)
        if file_path.endswith(".csv")
    ]
    assert all([os.path.exists(path) for path in file_paths])
    train_dataloader, val_dataloader = get_control_loaders(
        file_paths, batch_sizes=(train_val_batch_size, train_val_batch_size)
    )
    train_dataloader, val_dataloader = fabric.setup_dataloaders(
        train_dataloader, val_dataloader
    )

    # initialize wandb ==========================================
    if with_wandb:
        wandb.init(
            project="brains_neural_control",
            name=f"control|dpc",
            config=config,
        )
        wandb.watch(control_model, log_freq=1)

    # Run training loop with validation =========================================
    train(
        fabric,
        system_model,
        control_model,
        optimizer,
        train_dataloader,
        val_dataloader,
        control_output_ckpt=control_output_ckpt,
        scheduler=scheduler,
        num_epochs=num_epochs,
        loss_weights=loss_weights,
        with_wandb=with_wandb,
    )
    # save model ================================================================
    # fabric.save(control_output_ckpt, control_model.mlp.state_dict())
    # log the model to wandb
    if with_wandb:
        wandb.save(control_output_ckpt)

    # evaluate model on test set ================================================
    # load best model
    control_model.mlp.load_state_dict(
        torch.load(control_output_ckpt, map_location="cpu")
    )
    control_model.eval()
    # load test train_dataset
    file_paths = [
        os.path.abspath(os.path.join(test_data_dir, file_path))
        for file_path in os.listdir(test_data_dir)
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
    u0toNfminus1 = u0toNfminus1.unsqueeze(1).detach().cpu().numpy()
    x1toNf = x1toNf.unsqueeze(1).detach().cpu().numpy()
    if not os.path.exists("plots"):
        os.mkdir("plots")
    plot_names = [f"{system_model_name}_control_{i}.png" for i in range(5)]
    for i in range(5):
        plot_control_trajs(
            x0=x0[i],
            xref0toNf=xref0toNf[i],
            u0toNfminus1=u0toNfminus1[i],
            x1toNf=x1toNf[i],
            model_labels=[system_model_name + "_control"],
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
