# Copyright (c) 2023 Tudor Oancea
import argparse
import json
import os

import lightning as L
import scienceplots
import torch
import torch.nn.functional as F
from lightning import Fabric
from matplotlib import pyplot as plt

from math_591_project.models import *
from math_591_project.utils import *

L.seed_everything(127)
plt.style.use(["science"])
plt.rcParams.update({"font.size": 20})

Nfs = [1, 5, 10, 20]
# config_paths = [f"config/sysid/neural_dyn6_nf={i}.json" for i in Nfs] + [
#     "config/sysid/kin4.json"
# ]
# model_labels = [rf"NeuralDyn6 $N_f$={Nf}" for Nf in Nfs] + ["Kin4"]
config_paths = ["config/sysid/dyn6.json"]
model_labels = ["Dyn6"]


def dataset_velocity_distribution():
    train_data_dir = "dataset_v2.0.0/train_sysid"
    test_data_dir = "dataset_v2.0.0/test_sysid"
    train_file_paths = [
        os.path.abspath(os.path.join(train_data_dir, file_path))
        for file_path in os.listdir(train_data_dir)
        if file_path.endswith(".csv")
    ]
    test_file_paths = [
        os.path.abspath(os.path.join(test_data_dir, file_path))
        for file_path in os.listdir(test_data_dir)
        if file_path.endswith(".csv")
    ]
    assert all([os.path.exists(path) for path in train_file_paths])
    assert all([os.path.exists(path) for path in test_file_paths])
    x_train = np.vstack(
        [load_data(path, format="numpy")[1] for path in train_file_paths]
    )
    x_test = np.vstack([load_data(path, format="numpy")[1] for path in test_file_paths])

    print(
        f"number of data points:\n\ttrain: {x_train.shape[0]}\n\ttest: {x_test.shape[0]}"
    )

    v_x_train = x_train[:, 3]
    v_y_train = x_train[:, 4]
    r_train = x_train[:, 5]
    v_x_test = x_test[:, 3]
    v_y_test = x_test[:, 4]
    r_test = x_test[:, 5]
    plt.figure(figsize=(20, 10))
    plt.hist(
        v_x_train,
        bins=1000,
        range=(0, 13),
        color="blue",
        alpha=0.5,
        density=True,
        label="train",
    )
    plt.hist(
        v_x_test,
        bins=1000,
        range=(0, 13),
        color="red",
        alpha=0.5,
        density=True,
        label="test",
    )
    # plt.title(r"Distribution of $v_x$")
    plt.legend()
    plt.tight_layout()
    plt.savefig("experiments/sysid/v_x_distribution.png", dpi=300, bbox_inches="tight")

    plt.figure(figsize=(20, 10))
    plt.hist(
        v_y_train,
        bins=100,
        range=(-1, 1),
        color="blue",
        alpha=0.5,
        density=True,
        label="train",
    )
    plt.hist(
        v_y_test,
        bins=100,
        range=(-1, 1),
        color="red",
        alpha=0.5,
        density=True,
        label="test",
    )
    # plt.title(r"Distribution of $v_y$")
    plt.legend()
    plt.tight_layout()
    plt.savefig("experiments/sysid/v_y_distribution.png", dpi=300, bbox_inches="tight")

    plt.figure(figsize=(20, 10))
    plt.hist(
        r_train,
        bins=100,
        range=(-3, 3),
        color="blue",
        alpha=0.5,
        density=True,
        label="train",
    )
    plt.hist(
        r_test,
        bins=100,
        range=(-3, 3),
        color="red",
        alpha=0.5,
        density=True,
        label="test",
    )
    # plt.title(r"Distribution of $r$")
    plt.legend()
    plt.tight_layout()
    plt.savefig("experiments/sysid/r_distribution.png", dpi=300, bbox_inches="tight")


def load_config(config_path):
    return json.load(open(config_path, "r"))


def load_sysid_test_data(fabric: Fabric, config: dict):
    data_dir = config["testing"]["data_dir"]
    num_samples = config["testing"]["num_samples"]
    file_paths = [
        os.path.abspath(os.path.join(data_dir, file_path))
        for file_path in os.listdir(data_dir)
        if file_path.endswith(".csv")
    ]
    test_dataset = SysidTestDataset(file_paths, config["testing"]["Nf"])
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=num_samples if num_samples > 0 else len(test_dataset),
        shuffle=True,
        num_workers=1,
    )
    test_dataloader = fabric.setup_dataloaders(test_dataloader)

    return test_dataset, test_dataloader


def create_sysid_test_model(fabric: Fabric, config: dict):
    # model config
    model_config = config["model"]
    model_name = model_config["name"]
    ode_t = ode_from_string[model_name]
    model_is_neural = issubclass(ode_t, NeuralMixin)
    if model_is_neural:
        nhidden = model_config["n_hidden"]
        nonlinearity = model_config["nonlinearity"]
    input_checkpoint = model_config["input_checkpoint"]

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
        Nf=config["testing"]["Nf"],
    )

    # try:
    #     (
    #         system_model.model.ode.net if model_is_neural else system_model.model.ode
    #     ).load_state_dict(torch.load(input_checkpoint, map_location="cpu"))
    #     print("Successfully loaded model parameters from checkpoint")
    # except FileNotFoundError:
    #     print("No checkpoint found, using random initialization")
    # except RuntimeError:
    #     print("Checkpoint found, but not compatible with current model")

    system_model = fabric.setup(system_model)
    system_model.eval()
    system_model.requires_grad_(False)
    return system_model


def compute_sysid_errors(xtilde1toNf_p, xtilde1toNf, config):
    errors = torch.zeros((xtilde1toNf.shape[0], config["testing"]["Nf"], 5))
    errors[..., 0] = (
        F.mse_loss(xtilde1toNf_p[..., :2], xtilde1toNf[..., :2], reduction="none")
        .sum(dim=-1)
        .sqrt()
    )
    errors[..., 1] = F.l1_loss(
        xtilde1toNf_p[..., 2], xtilde1toNf[..., 2], reduction="none"
    )
    errors[..., 2] = F.l1_loss(
        xtilde1toNf_p[..., 3], xtilde1toNf[..., 3], reduction="none"
    )
    if xtilde1toNf_p.shape[-1] > 4:
        errors[..., 3] = F.l1_loss(
            xtilde1toNf_p[..., 4], xtilde1toNf[..., 4], reduction="none"
        )
        errors[..., 4] = F.l1_loss(
            xtilde1toNf_p[..., 5], xtilde1toNf[..., 5], reduction="none"
        )

    return errors.cpu().numpy()


@torch.no_grad()
def sysid_errors():
    fabric = Fabric()
    errors = []
    for i, config_path in enumerate(config_paths):
        print(f"Computing errors for {model_labels[i]}...")
        config = load_config(config_path)
        _, test_dataloader = load_sysid_test_data(fabric, config)
        system_model = create_sysid_test_model(fabric, config)
        xtilde0, utilde0toNfminus1, xtilde1toNf = next(iter(test_dataloader))
        xtilde1toNf_p = system_model(
            xtilde0 if i < len(Nfs) else xtilde0[..., :4], utilde0toNfminus1
        )

        # compute the test errors (XY, phi, v_x, v_y, r) for each stage (1 to Nf) and for each sample
        # => errors has shape (dataset size, Nf, 5)
        errors.append(compute_sysid_errors(xtilde1toNf_p, xtilde1toNf, config))

    # compute means by stage and by variable => list of arrays of shape (Nf, 5)
    means = [error.mean(axis=0) for error in errors]
    stds = [error.std(axis=0) for error in errors]
    colors = ["blue", "orange", "green", "red", "purple", "brown"]
    plot_titles = ["XY", "phi", "v_x", "v_y", "r"]
    for i in range(len(plot_titles)):
        plt.figure(figsize=(10, 5))
        x = np.arange(1, 1 + config["testing"]["Nf"])
        for j, (mean, std) in enumerate(zip(means, stds)):
            if j >= 4 and i >= 3:
                continue
            plt.plot(
                x,
                mean[:, i],
                color=colors[j],
                label=model_labels[j],
            )

        plt.xlabel("stage")
        plt.ylabel("error")
        plt.title(plot_titles[i])
        plt.legend()
        plt.tight_layout()
        # plt.savefig(
        #     f"experiments/sysid/sysid_errors_{plot_titles[i]}.png",
        #     dpi=300,
        #     bbox_inches="tight",
        # )
        # plt.show

    # compute means by variable => list of arrays of shape (5,)
    means = [error.mean(axis=(0, 1)) for error in errors]
    stds = [error.std(axis=(0, 1)) for error in errors]
    # print the results in the form mean std by variable and by model
    for i, (mean, std) in enumerate(zip(means, stds)):
        print(
            f"{model_labels[i]}: XY:{mean[0]:.4f}±{std[0]:.4f}, phi:{mean[1]:.4f}±{std[1]:.4f}, v_x:{mean[2]:.4f}±{std[2]:.4f}, v_y:{mean[3]:.4f}±{std[3]:.4f}, r:{mean[4]:.4f}±{std[4]:.4f}"
        )


def sysid_losses():
    losses = {}
    for Nf in Nfs:
        data = pd.read_csv(f"experiments/sysid/neural_dyn6_nf={Nf}.csv")
        losses[Nf] = {
            "train": data["train_loss"].values[:300],
            "validation": data["val_loss"].values[:300],
        }

    plt.figure(figsize=(20, 10))
    colors = ["blue", "orange", "green", "red", "purple", "brown"]
    for i, Nf in enumerate(Nfs):
        plt.plot(
            losses[Nf]["train"],
            label=rf"train $N_f$={Nf}",
            color=colors[i],
            linestyle="-",
        )
        plt.plot(
            losses[Nf]["validation"],
            label=rf"validation $N_f$={Nf}",
            color=colors[i],
            linestyle="--",
        )
    plt.legend()
    plt.tight_layout()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig("experiments/sysid/sysid_losses.png", dpi=300, bbox_inches="tight")


@torch.no_grad()
def sysid_trajs():
    fabric = Fabric()
    test_dataset, test_dataloader = load_sysid_test_data(
        fabric, load_config(config_paths[0])
    )
    # id = torch.randint(0, len(test_dataset), (1,)).item()
    id = 7800
    print("using id:", id)
    xtilde0, utilde0toNfminus1, xtilde1toNf = test_dataset[id]
    xtilde0, utilde0toNfminus1, xtilde1toNf = (
        xtilde0.unsqueeze(0).to(fabric.device),
        utilde0toNfminus1.unsqueeze(0).to(fabric.device),
        xtilde1toNf.unsqueeze(0).to(fabric.device),
    )
    xtilde1toNf_ps = []

    for i, config_path in enumerate(config_paths):
        config = load_config(config_path)
        system_model = create_sysid_test_model(fabric, config)
        system_model.eval()
        xtilde1toNf_p = system_model(
            xtilde0 if i < len(Nfs) else xtilde0[..., :4], utilde0toNfminus1
        )  # shape (1, Nf, 6) or (1, Nf, 4)
        xtilde1toNf_ps.append(
            xtilde1toNf_p
            if i < len(Nfs)
            else torch.cat(
                [
                    xtilde1toNf_p,
                    torch.full(
                        (1, xtilde1toNf_p.shape[1], 2), np.nan, device=fabric.device
                    ),
                ],
                dim=-1,
            )
        )

    xtilde1toNf_ps = torch.cat(xtilde1toNf_ps, dim=0)  # shape (len(Nfs), Nf, 6)

    plot_sysid_trajs(
        xtilde0=xtilde0.squeeze(0).cpu().numpy(),
        utilde0toNfminus1=utilde0toNfminus1.squeeze(0).cpu().numpy(),
        xtilde1toNf=xtilde1toNf.squeeze(0).cpu().numpy(),
        xtilde1toNf_p=xtilde1toNf_ps.cpu().numpy(),
        model_labels=model_labels,
        dt=1 / 20,
    )
    # plt.savefig(f"experiments/sysid/sysid_trajs.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    # print("Dataset velocity distribution =====================")
    # dataset_velocity_distribution()
    # print("Sysid losses =======================================")
    # sysid_losses()
    # print("Sysid errors =======================================")
    # sysid_errors()
    print("Sysid trajs ========================================")
    sysid_trajs()
