# Copyright (c) 2023 Tudor Oancea
import argparse
import json
import os
from pyexpat import model

import lightning as L
import torch
import torch.nn.functional as F
from lightning import Fabric
from matplotlib import pyplot as plt

from math_591_project.utils.data_utils import *
from math_591_project.models import *
from math_591_project.utils.plot_utils import *
from math_591_project.utils.plot_utils import plot_open_loop_predictions

L.seed_everything(127)
plt.style.use(["science"])
plt.rcParams.update({"font.size": 20})


def load_config(config_path):
    return json.load(open(config_path, "r"))


def load_sysid_test_data(fabric: Fabric, config: dict):
    data_dir = os.path.join(config["data"]["dir"], config["data"]["test"])
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
    model_name = config["model"]["name"]
    is_blackbox = model_name.startswith("blackbox")
    dims = ode_dims[model_name]
    if is_blackbox:
        ode_t, nxtilde, nutilde, nin, nout = dims
    else:
        ode_t, nxtilde, nutilde = dims

    system_model = OpenLoop(
        model=RK4(
            nxtilde=nxtilde,
            nutilde=nutilde,
            ode=ode_t(
                net=MLP(
                    nin=nin,
                    nout=nout,
                    nhidden=config["model"]["n_hidden"],
                    nonlinearity=config["model"]["nonlinearity"],
                )
            )
            if is_blackbox
            else ode_t(),
            dt=1 / 20,
        ),
        Nf=config["testing"]["Nf"],
    )
    try:
        system_model.model.ode.load_state_dict(
            torch.load(config["model"]["checkpoint_path"], map_location="cpu")[
                "system_model"
            ]
        )
        # print("Successfully loaded model parameters from checkpoint")
    except FileNotFoundError:
        print("No checkpoint found, using random initialization")
    except RuntimeError:
        print("Checkpoint found, but not compatible with current model")

    system_model = fabric.setup(system_model)

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
def old_main():
    parser = argparse.ArgumentParser(description="arg parser")
    parser.add_argument(
        "--cfg_file",
        type=str,
        default="config/blackbox_dyn6_sysid.json",
        help="specify the config for testing",
    )
    args = parser.parse_args()

    # load config from specified json file ============================================
    config = load_config(args.cfg_file)

    fabric = Fabric()
    print(f"Using {fabric.device} device")
    system_model = create_sysid_test_model(fabric, config)

    # load test data ==================================================================
    test_dataset, test_dataloader = load_sysid_test_data(fabric, config)

    # run model on test set ===========================================================
    system_model.eval()
    xtilde0, utilde0toNfminus1, xtilde1toNf = next(iter(test_dataloader))
    xtilde1toNf_p = system_model(xtilde0, utilde0toNfminus1)

    # compute the test losses (XY, phi, v_x, v_y, r) for each stage (1 to Nf) and for each sample
    # => losses has shape (dataset size, Nf, 5)
    errors = compute_sysid_errors(xtilde1toNf_p, xtilde1toNf, config)

    # compute means by variable
    # => means has shape (5,)
    means = errors.mean(axis=(0, 1))
    stds = errors.std(axis=(0, 1))
    print(
        f"Mean errors: XY: {means[0]:.3f}±{stds[0]:.3f}, phi: {means[1]:.3f}±{stds[0]:.3f}, v_x: {means[2]:.3f}±{stds[2]:.3f}, v_y: {means[3]:.3f}±{stds[3]:.3f}, r: {means[4]:.3f}±{stds[4]:.3f}"
    )

    # compute means by stage and by variable
    # => means has shape (Nf, 5)
    means = errors.mean(axis=0)
    stds = errors.std(axis=0)
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    for i, ax in enumerate(axes):
        up = means[:, i] + stds[:, i]
        down = means[:, i] - stds[:, i]
        x = np.arange(1, 1 + config["testing"]["Nf"])
        ax.plot(x, means[:, i], color="blue", label="mean")
        ax.plot(x, up, color="blue", linestyle="-.", label="mean+std")
        ax.plot(x, down, color="blue", linestyle="-.", label="mean-std")
        ax.fill_between(x, up, down, color="blue", alpha=0.2)
        ax.set_xlabel("stage")
        ax.set_ylabel("error")
        ax.set_title(["XY", "phi", "v_x", "v_y", "r"][i])
    fig.suptitle(args.cfg_file.split("/")[-1].split(".")[0])
    plt.tight_layout()
    plt.savefig(
        f'{args.cfg_file.split("/")[-1].split(".")[0]}', dpi=300, bbox_inches="tight"
    )

    # plot the loss distributions for each stage as a scatter plot (one plot per sample)
    # fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    # for i, ax in enumerate(axes):
    #     ax.scatter(
    #         np.kron(
    #             np.arange(1, 1 + config["testing"]["Nf"]), np.ones(xtilde1toNf.shape[0])
    #         ),
    #         errors[..., i].flatten(),
    #         s=1,
    #     )
    #     ax.set_title(["XY", "phi", "v_x", "v_y", "r"][i])

    # plt.tight_layout()
    # plt.savefig("loss_distributions.png", dpi=300, bbox_inches="tight")
    # plt.show()

    # compute the test loss
    # XY_loss = F.mse_loss(xtilde1toNf_p[..., :2], xtilde1toNf[..., :2])
    # phi_loss = F.mse_loss(xtilde1toNf_p[..., 2], xtilde1toNf[..., 2])
    # v_x_loss = F.mse_loss(xtilde1toNf_p[..., 3], xtilde1toNf[..., 3])
    # if xtilde1toNf_p.shape[-1] > 4:
    #     v_y_loss = F.mse_loss(xtilde1toNf_p[..., 4], xtilde1toNf[..., 4])
    #     r_loss = F.mse_loss(xtilde1toNf_p[..., 5], xtilde1toNf[..., 5])
    # else:
    #     v_y_loss = torch.tensor(0.0)
    #     r_loss = torch.tensor(0.0)
    # print(
    #     f"test losses:\n\tXY: {XY_loss:.4f}\n\tphi: {phi_loss:.4f}\n\tv_x: {v_x_loss:.4f}\n\tv_y: {v_y_loss:.4f}\n\tr: {r_loss:.4f}"
    # )

    # move all tensors to the cpu and plot them
    # if num_samples > 0:
    #     xtilde0 = xtilde0.detach().cpu().numpy()
    #     utilde0toNfminus1 = utilde0toNfminus1.detach().cpu().numpy()
    #     xtilde1toNf = xtilde1toNf.detach().cpu().numpy()
    #     xtilde1toNf_p = xtilde1toNf_p.detach().cpu().numpy()
    #     if not os.path.exists("test_plots"):
    #         os.mkdir("test_plots")
    #     plot_names = [
    #         f"test_plots/{model_name}_{checkpoint_path.split('/')[-1].split('.')[0]}_{i}.png"
    #         for i in range(num_samples)
    #     ]
    #     for i in range(num_samples):
    #         (plot_kin4 if model_name.endswith("kin4") else plot_dyn6)(
    #             xtilde0=xtilde0[i],
    #             utilde0toNfminus1=utilde0toNfminus1[i],
    #             xtilde1toNf=xtilde1toNf[i],
    #             xtilde1toNf_p=xtilde1toNf_p[i],
    #             dt=dt,
    #         )
    #         plt.savefig(plot_names[i], dpi=300)


@torch.no_grad()
def plot_errors():
    Nfs = [1, 5, 10, 20]
    config_paths = [f"config/test_sysid/nf{i}.json" for i in Nfs] + [
        "config/test_sysid/kin4.json"
    ]
    _, test_dataloader = load_sysid_test_data(fabric, config)
    fabric = Fabric()
    errors = []
    for i, config_path in enumerate(config_paths):
        print("Computing errors for", f"Nf={Nfs[i]}" if i < len(Nfs) else "kin4")
        config = load_config(config_path)
        system_model = create_sysid_test_model(fabric, config)
        system_model.eval()
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
                label=rf"$N_f$={Nfs[j]}" if j < 4 else "Kin4",
            )
            # plt.plot(x, mean[:, i] + 3 * std[:, i], color=colors[j], linestyle="-.")
            # plt.plot(x, mean[:, i] - 3 * std[:, i], color=colors[j], linestyle="-.")

        plt.xlabel("stage")
        plt.legend()
        plt.ylabel("error")
        # plt.title(rf"${plot_titles[i]}$")
        plt.tight_layout()
        plt.savefig(f"sysid_errors_{plot_titles[i]}.png", dpi=300, bbox_inches="tight")

    # compute means by variable => list of arrays of shape (5,)
    means = [error.mean(axis=(0, 1)) for error in errors]
    stds = [error.std(axis=(0, 1)) for error in errors]
    # print the results in the form mean std by variable and by model
    for i, (mean, std) in enumerate(zip(means, stds)):
        title = f"nf={Nfs[i]}" if i < 4 else "kin4"
        print(
            f"{title}: XY:{mean[0]:.4f}±{std[0]:.4f}, phi:{mean[1]:.4f}±{std[1]:.4f}, v_x:{mean[2]:.4f}±{std[2]:.4f}, v_y:{mean[3]:.4f}±{std[3]:.4f}, r:{mean[4]:.4f}±{std[4]:.4f}"
        )


def dataset_velocity_distribution():
    train_data_dir = "data_v1.1.0_sysid/train"
    test_data_dir = "data_v1.1.0_sysid/test"
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
    plt.savefig("v_x_distribution.png", dpi=300, bbox_inches="tight")

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
    plt.savefig("v_y_distribution.png", dpi=300, bbox_inches="tight")

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
    plt.savefig("r_distribution.png", dpi=300, bbox_inches="tight")


def plot_losses():
    nfs = [1, 5, 10, 20]
    losses = {}
    for nf in nfs:
        data = pd.read_csv(f"nf={nf}.csv")
        losses[nf] = {
            "train": data["train_loss"].values[:300],
            "validation": data["val_loss"].values[:300],
        }

    plt.figure(figsize=(20, 10))
    colors = ["blue", "orange", "green", "red", "purple", "brown"]
    for i, nf in enumerate(nfs):
        plt.plot(
            losses[nf]["train"],
            label=rf"train $N_f$={nf}",
            color=colors[i],
            linestyle="-",
        )
        plt.plot(
            losses[nf]["validation"],
            label=rf"validation $N_f$={nf}",
            color=colors[i],
            linestyle="--",
        )
    plt.legend()
    plt.tight_layout()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig("sysid_losses.png", dpi=300, bbox_inches="tight")
    # plt.show()


@torch.no_grad()
def plot_trajs():
    Nfs = [1, 5, 10, 20]
    config_paths = [f"config/test_sysid/nf{i}.json" for i in Nfs] + [
        "config/test_sysid/kin4.json"
    ]
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

    plot_open_loop_predictions(
        xtilde0=xtilde0.squeeze(0).cpu().numpy(),
        utilde0toNfminus1=utilde0toNfminus1.squeeze(0).cpu().numpy(),
        xtilde1toNf=xtilde1toNf.squeeze(0).cpu().numpy(),
        xtilde1toNf_p=xtilde1toNf_ps.cpu().numpy(),
        model_labels=[rf"$N_f$={nf}" for nf in Nfs] + ["Kin4"],
        dt=1 / 20,
    )
    plt.savefig(f"sysid_trajs.png", dpi=300)


if __name__ == "__main__":
    # main()
    # plot_errors()
    # dataset_velocity_distribution()
    # plot_losses()
    plot_trajs()
