# Copyright (c) 2023 Tudor Oancea
import os
from copy import copy

import lightning as L
import torch
import torch.nn.functional as F
import wandb
from lightning import Fabric
from matplotlib import pyplot as plt
from math_591_project.data_utils import *
from math_591_project.models import *
from math_591_project.plot_utils import *
from tqdm import tqdm

L.seed_everything(127)
with_wandb = True


def run_model(model, batch):
    x0, xref0toNf, uref0toNfminus1 = batch
    u0toNfminus1 = model(x0, xref0toNf)
    T_loss = F.mse_loss(u0toNfminus1[:, :, 0], uref0toNfminus1[:, :, 0])
    ddelta_loss = F.mse_loss(u0toNfminus1[:, :, 1], uref0toNfminus1[:, :, 1])
    return T_loss, ddelta_loss


def train(
    fabric,
    control_model,
    optimizer,
    train_dataloader,
    val_dataloader,
    scheduler=None,
    num_epochs=10,
    loss_weights=(1.0, 1.0),
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
            "train_T_loss": train_T_loss,
            "train_ddelta_loss": train_ddelta_loss,
            "val_T_loss": val_T_loss,
            "val_ddelta_loss": val_ddelta_loss,
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
                f"checkpoints/kin4_control_imitation_best.ckpt",
                {
                    "control_model": control_model.mlp,
                    "optimizer": optimizer,
                    "scheduler": scheduler,
                },
            )


def main():
    torch.autograd.set_detect_anomaly(True)
    # set up config =========================================================
    config = {
        "model": {
            "n_hidden": (256, 256),
            "nonlinearity": "relu",
            "from_checkpoint": False,
        },
        "training": {
            "num_epochs": 1000,
            "loss_weights": (1.0, 1.0),
            "optimizer": {
                "name": "adam",
                "lr": 1e-3,
            },
            "scheduler": {
                "name": "none",
                # "name": "multisteplr",
                # "milestones": [701, 1601],
                # "gamma": 0.5,
            },
        },
        "data": {
            "dataset": "control1",
        },
    }
    n_hidden = config["model"]["n_hidden"]
    nonlinearity = config["model"]["nonlinearity"]
    from_checkpoint = config["model"]["from_checkpoint"]
    num_epochs = config["training"]["num_epochs"]
    optimizer_params = config["training"]["optimizer"]
    optimizer = optimizer_params.pop("name")
    scheduler_params = config["training"]["scheduler"]
    scheduler = scheduler_params.pop("name")
    dataset = config["data"]["dataset"]

    # initialize wandb ==========================================
    if with_wandb:
        wandb.init(
            project="brains_neural_control",
            name=f"control|imitation",
            config=config,
        )

    # intialize lightning fabric ===============================================
    fabric = Fabric()

    # i_nitialize model and optimizer =========================================================
    control_model = MLPControlPolicy(
        nx=5,
        nu=2,
        Nf=20,
        mlp=MLP(
            nin=5 + 21 * 4, nout=2 * 20, nhidden=n_hidden, nonlinearity=nonlinearity
        ),
    )
    if from_checkpoint:
        control_model.mlp.load_state_dict(
            torch.load("checkpoints/kin4_control_imitation_best.ckpt")[
                "control_mlp"
            ].mlp.state_dict()
        )
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
    data_dir = "data/control/" + dataset
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
        control_model,
        optimizer,
        train_dataloader,
        val_dataloader,
        scheduler=scheduler,
        num_epochs=num_epochs,
        loss_weights=(1.0, 1.0),
    )

    # save model ================================================================
    fabric.save(
        f"checkpoints/kin4_control_imitation_final.ckpt",
        {"model": control_model.mlp},
    )

    if with_wandb:
        # log the model to wandb
        wandb.save(f"checkpoints/kin4_control_imitation_final.ckpt")
        wandb.save(f"checkpoints/kin4_control_imitation_best.ckpt")

    # evaluate model on test set ================================================
    # recreate open loop model with new Nf
    # model = OpenLoop(
    #     model=RK4(
    #         nxtilde=nxtilde,
    #         nutilde=nutilde,
    #         ode=(BlackboxKin4ODE if base_model == "kin4" else BlackboxDyn6ODE)(
    #             net=MLP(
    #                 nin=nxtilde + nutilde
    #                 if base_model == "kin4"
    #                 else nxtilde - 3 + nutilde,
    #                 nout=nxtilde if base_model == "kin4" else nxtilde - 3,
    #                 nhidden=n_hidden,
    #                 nonlinearity=nonlinearity,
    #             )
    #         ),
    #         dt=dt,
    #     ),
    #     Nf=Nf,
    # )
    # model.load_state_dict(torch.load(f"checkpoints/{base_model}_best.ckpt")["model"])
    # model = fabric.setup(model)

    # evaluate model on test set
    # model.eval()
    # x0, Uf, Xf = next(iter(val_dataloader))
    # Xfpred = model(x0, Uf)
    # id = torch.randint(0, x0.shape[0], (5,))
    # x0 = x0[id].detach().cpu().numpy()
    # Uf = Uf[id].detach().cpu().numpy()
    # Xf = Xf[id].detach().cpu().numpy()
    # Xfpred = Xfpred[id].detach().cpu().numpy()
    # if not os.path.exists("plots"):
    #     os.mkdir("plots")
    # plot_names = [f"plots/{base_model}_{i}.png" for i in range(5)]
    # for i in range(5):
    #     (plot_kin4 if base_model == "kin4" else plot_dyn6)(
    #         x0=x0[i], Uf=Uf[i], Xf=Xf[i], Xfpred=Xfpred[i], dt=dt
    #     )
    #     plt.savefig(plot_names[i], dpi=300)
    # if with_wandb:
    #     # log the plot to wandb
    #     wandb.log(
    #         {"plot/" + plot_name: wandb.Image(plot_name) for plot_name in plot_names}
    #     )

    # plt.show()


if __name__ == "__main__":
    main()
