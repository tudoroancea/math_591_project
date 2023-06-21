# Copyright (c) 2023 Tudor Oancea
import argparse
import json
import os
from copy import copy
from pyexpat import model

import lightning as L
import torch
import torch.nn.functional as F
import torch._dynamo
import wandb
from lightning import Fabric
from matplotlib import pyplot as plt
from tqdm import tqdm

from math_591_project.utils import *
from math_591_project.models import *

import logging

log = logging.getLogger("lightning")
log.propagate = False
log.setLevel(logging.ERROR)

L.seed_everything(127)


def run_model(system_model, batch, ode_t):
    xtilde0, utilde0toNfminus1, xtilde1toNf = batch
    xtilde1toNf_p = system_model(
        xtilde0[..., :4] if ode_t == Kin4 else xtilde0, utilde0toNfminus1
    )
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
    ode_t: type[Kin4],
    system_model: OpenLoop,
    optimizer: torch.optim.Optimizer,
    train_dataloader: torch.utils.data.DataLoader,
    val_dataloader: torch.utils.data.DataLoader,
    output_checkpoint: str,
    scheduler: torch.optim.lr_scheduler.LRScheduler = None,
    num_epochs=10,
    loss_weights={"XY": 1.0, "phi": 1.0, "v_x": 1.0, "v_y": 1.0, "r": 1.0},
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
                system_model, batch, ode_t
            )
            loss = (
                loss_weights["XY"] * XY_loss
                + loss_weights["phi"] * phi_loss
                + loss_weights["v_x"] * v_x_loss
                + loss_weights["v_y"] * v_y_loss
                + loss_weights["r"] * r_loss
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
                    system_model, batch, ode_t
                )
            val_losses["total"] += (
                loss_weights["XY"] * XY_loss
                + loss_weights["phi"] * phi_loss
                + loss_weights["v_x"] * v_x_loss
                + loss_weights["v_y"] * v_y_loss
                + loss_weights["r"] * r_loss
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
                output_checkpoint,
                system_model.model.ode.state_dict(),
            )


def main():
    parser = argparse.ArgumentParser(description="arg parser")
    parser.add_argument(
        "--cfg_file",
        type=str,
        default="config/sysid/neuraldyn6_nf10.json",
        help="specify the config file used for training",
    )
    args = parser.parse_args()

    # extract config =================================================================
    ic(args.cfg_file)
    config = json.load(open(args.cfg_file, "r"))
    with_wandb = config.pop("with_wandb")
    print(f"Training " + ("with" if with_wandb else "without") + " wandb")

    # model config
    model_config = config["model"]
    model_name: str = model_config["name"]
    ode_t = ode_from_string[model_name]
    model_is_neural = issubclass(ode_t, NeuralMixin)
    if model_is_neural:
        nhidden = model_config["n_hidden"]
        nonlinearity = model_config["nonlinearity"]
    from_checkpoint = model_config["from_checkpoint"]
    input_checkpoint = model_config["input_checkpoint"]
    output_checkpoint = model_config["output_checkpoint"]

    # training config
    train_config = config["training"]
    train_Nf = train_config["Nf"]
    num_epochs = train_config["num_epochs"]
    train_val_batch_size = train_config["batch_size"]
    loss_weights = train_config["loss_weights"]
    train_data_dir = train_config["data_dir"]

    optimizer_config = train_config["optimizer"]
    optimizer = optimizer_config.pop("name")

    scheduler_config = train_config["scheduler"]
    scheduler = scheduler_config.pop("name")

    test_config = config["testing"]
    test_data_dir = test_config["data_dir"]
    test_Nf = test_config["Nf"]
    test_batch_size = (
        test_config["num_samples"] if test_config["num_samples"] > 0 else 5
    )

    # intialize lightning fabric ===============================================
    fabric = Fabric()
    print("Using device: ", fabric.device)

    # initialize model and optimizer =========================================================
    ode_t = ode_from_string[model_name]

    system_model = OpenLoop(
        model=RK4(
            ode=ode_t(
                net=MLP(
                    nin=ode_t.nin,
                    nout=ode_t.nout,
                    nhidden=nhidden,
                    nonlinearity=nonlinearity,
                )
            )
            if model_is_neural
            else ode_t(),
            dt=dt,
        ),
        Nf=train_Nf,
    )
    if with_wandb:
        wandb.watch(system_model, log_freq=1)

    if from_checkpoint:
        try:
            (
                system_model.model.ode.net
                if model_is_neural
                else system_model.model.ode
            ).load_state_dict(torch.load(input_checkpoint, map_location="cpu"))
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
        case _:
            raise NotImplementedError(f"Optimizer {optimizer} not implemented")

    optimizer = optimizer_t(system_model.parameters(), **optimizer_config)

    match scheduler:
        case "steplr":
            scheduler_t = torch.optim.lr_scheduler.StepLR
        case "multisteplr":
            scheduler_t = torch.optim.lr_scheduler.MultiStepLR
        case _:
            scheduler_t = None

    scheduler = scheduler_t(optimizer, **scheduler_config) if scheduler_t else None

    system_model, optimizer = fabric.setup(system_model, optimizer)

    # load data ================================================================
    # load all CSV files in data/sysid
    file_paths = [
        os.path.abspath(os.path.join(train_data_dir, file_path))
        for file_path in os.listdir(train_data_dir)
        if file_path.endswith(".csv")
    ]
    assert all([os.path.exists(path) for path in file_paths])
    train_dataloader, val_dataloader = get_sysid_loaders(
        file_paths,
        batch_sizes=(train_val_batch_size, train_val_batch_size),
        Nf=train_Nf,
    )
    train_dataloader, val_dataloader = fabric.setup_dataloaders(
        train_dataloader, val_dataloader
    )

    # initialize wandb ==========================================
    if with_wandb:
        wandb.init(
            project="brains_neural_control",
            name=f"sysid|{model_name}",
            config=config,
        )

    # Run training loop with validation =========================================
    try:
        train(
            fabric,
            ode_t,
            system_model,
            optimizer,
            train_dataloader,
            val_dataloader,
            output_checkpoint=output_checkpoint,
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
    # fabric.save(output_checkpoint, system_model.model.ode.state_dict())
    # log the model to wandb
    if with_wandb:
        wandb.save(output_checkpoint)

    # evaluate model on test set ================================================
    # recreate open loop model with new Nf
    system_model = OpenLoop(
        model=RK4(
            ode=ode_t(
                net=MLP(
                    nin=ode_t.nin,
                    nout=ode_t.nout,
                    nhidden=nhidden,
                    nonlinearity=nonlinearity,
                )
            )
            if model_is_neural
            else ode_t(),
            dt=dt,
        ),
        Nf=test_Nf,
    )
    try:
        system_model.model.ode.load_state_dict(
            torch.load(output_checkpoint, map_location="cpu")
        )
        print("Successfully loaded model parameters from checkpoint for testing")
    except FileNotFoundError:
        system_model.model.ode.load_state_dict(
            torch.load(input_checkpoint, map_location="cpu")
        )
    except RuntimeError:
        print("Checkpoint found, but not compatible with current model")

    system_model = fabric.setup(system_model)

    # create test dataloader
    file_paths = [
        os.path.abspath(os.path.join(test_data_dir, file_path))
        for file_path in os.listdir(test_data_dir)
        if file_path.endswith(".csv")
    ]
    test_dataset = SysidTestDataset(file_paths, test_Nf)
    test_dataloader = DataLoader(
        test_dataset, batch_size=test_batch_size, shuffle=True, num_workers=1
    )
    test_dataloader = fabric.setup_dataloaders(test_dataloader)

    # evaluate model on test set
    system_model.eval()
    xtilde0, utilde0toNfminus1, xtilde1toNf = next(iter(test_dataloader))
    xtilde1toNf_p = system_model(
        xtilde0[..., :4] if ode_t == Kin4 else xtilde0,
        utilde0toNfminus1,
    )
    if ode_t == Kin4:
        # add nans to xtilde0
        xtilde0 = torch.cat(
            [
                xtilde0[..., :4],
                torch.full_like(xtilde0[..., :2], fill_value=np.nan),
            ],
            dim=-1,
        )
        xtilde1toNf_p = torch.cat(
            [
                xtilde1toNf_p[..., :4],
                torch.full_like(xtilde1toNf_p[..., :2], fill_value=np.nan),
            ],
            dim=-1,
        )

    xtilde0 = xtilde0.detach().cpu().numpy()  # shape (5, 1, 6)
    utilde0toNfminus1 = utilde0toNfminus1.detach().cpu().numpy()  # shape (5, Nf, 2)
    xtilde1toNf = xtilde1toNf.detach().cpu().numpy()  # shape (5, Nf, 6)
    xtilde1toNf_p = (
        xtilde1toNf_p.unsqueeze(1).detach().cpu().numpy()
    )  # shape (5, 1, Nf, 6)
    if not os.path.exists("plots"):
        os.mkdir("plots")
    plot_names = [f"{model_name}_{i}.png" for i in range(5)]
    for i in range(5):
        plot_open_loop_predictions(
            xtilde0=xtilde0[i],
            utilde0toNfminus1=utilde0toNfminus1[i],
            xtilde1toNf=xtilde1toNf[i],
            xtilde1toNf_p=xtilde1toNf_p[i],
            model_labels=[model_name],
            dt=dt,
        )
        plt.savefig("plots/" + plot_names[i], dpi=300)
    # log the plots to wandb
    if with_wandb:
        wandb.log(
            {
                "plot/" + plot_name: wandb.Image("plots/" + plot_name)
                for plot_name in plot_names
            }
        )

    plt.show()


if __name__ == "__main__":
    main()
