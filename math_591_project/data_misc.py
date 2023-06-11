import os
import shutil

import matplotlib.pyplot as plt
import numpy as np

from math_591_project.data_utils import *

np.random.seed(127)


def dataset_velocity_distribution():
    train_data_dir = "data/train"
    test_data_dir = "data/test"
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
    plt.figure(figsize=(10, 10))
    plt.hist(v_x_train, bins=1000, range=(0, 13), color="blue", alpha=0.5, density=True)
    plt.hist(v_x_test, bins=1000, range=(0, 13), color="red", alpha=0.5, density=True)
    plt.title(r"Distribution of $v_x$")
    plt.tight_layout()
    plt.savefig("v_x_distribution.png", dpi=300, bbox_inches="tight")

    plt.figure(figsize=(10, 10))
    plt.hist(v_y_train, bins=100, range=(-1, 1), color="blue", alpha=0.5, density=True)
    plt.hist(v_y_test, bins=100, range=(-1, 1), color="red", alpha=0.5, density=True)
    plt.title(r"Distribution of $v_y$")
    plt.tight_layout()
    plt.savefig("v_y_distribution.png", dpi=300, bbox_inches="tight")

    plt.figure(figsize=(10, 10))
    plt.hist(r_train, bins=100, range=(-3, 3), color="blue", alpha=0.5, density=True)
    plt.hist(r_test, bins=100, range=(-3, 3), color="red", alpha=0.5, density=True)
    plt.title(r"Distribution of $r$")
    plt.tight_layout()
    plt.savefig("r_distribution.png", dpi=300, bbox_inches="tight")


def load_old_data(file_path):
    df = pd.read_csv(file_path)
    x_cols = ["X", "Y", "phi", "v_x", "v_y", "r"]
    u_cols = ["T", "delta"]
    # remove points where the car is not movinv (v_x < 0.5)
    df = df[df["v_x"] > 0.01]
    timestamp = df["timestamp"].to_numpy()
    x = df[x_cols].to_numpy()
    u = df[u_cols].to_numpy()
    return timestamp, x, u


def portesouvertes_dataset_velocity_distribution():
    file_paths = [
        os.path.abspath(os.path.join("data_portes_ouvertes", file_path))
        for file_path in os.listdir("data_portes_ouvertes")
        if file_path.endswith(".csv")
    ]
    x = [load_old_data(path)[1] for path in file_paths]
    x = np.vstack(x)

    print(f"Number of data points: {x.shape[0]}")

    v_x = x[:, 3]
    v_y = x[:, 4]
    r = x[:, 5]

    plt.figure(figsize=(10, 10))
    plt.hist(v_x, bins=1000, range=(0, 13), color="blue", alpha=1.0, density=True)
    plt.title(r"Distribution of $v_x$")
    plt.tight_layout()
    plt.savefig("v_x_distribution_portes_ouvertes.png", dpi=300, bbox_inches="tight")

    plt.figure(figsize=(10, 10))
    plt.hist(v_y, bins=100, range=(-1, 1), color="blue", alpha=1.0, density=True)
    plt.title(r"Distribution of $v_y$")
    plt.tight_layout()
    plt.savefig("v_y_distribution_portes_ouvertes.png", dpi=300, bbox_inches="tight")

    plt.figure(figsize=(10, 10))
    plt.hist(r, bins=100, range=(-3, 3), color="blue", alpha=1.0, density=True)
    plt.title(r"Distribution of $r$")
    plt.tight_layout()
    plt.savefig("r_distribution_portes_ouvertes.png", dpi=300, bbox_inches="tight")


def preprocess_data_portes_ouvertes():
    """
    Loads data in the old format and saves it in the new format:
    timestamp,X,Y,phi,v_x,v_y,r,last_delta,X_ref_0,Y_ref_0,phi_ref_0,v_x_ref_0,X_ref_1,...,v_x_ref_40,T_0,ddelta_0,T_1,ddelta_1,...T_39,ddelta_39

    where ddelta corresponds to the difference between the last_delta and the new_delta.
    we can assume the first last_delta will be 0 and put 0 for the reference values as well as for the future predictions of T and ddelta.
    """
    output_dir = "data_po"
    os.makedirs(output_dir, exist_ok=True)

    header = "timestamp,X,Y,phi,v_x,v_y,r,last_delta"
    for i in range(41):
        header += f",X_ref_{i},Y_ref_{i},phi_ref_{i},v_x_ref_{i}"
    for i in range(40):
        header += f",T_{i},ddelta_{i}"

    data_dir = "processed"
    file_paths = [
        os.path.abspath(os.path.join(data_dir + "/samedi", file_path))
        for file_path in os.listdir(data_dir + "/samedi")
        if file_path.endswith(".csv")
    ] + [
        os.path.abspath(os.path.join(data_dir + "/dimanche", file_path))
        for file_path in os.listdir(data_dir + "/dimanche")
        if file_path.endswith(".csv")
    ]
    assert all([os.path.exists(path) for path in file_paths])
    new_file_paths = [
        os.path.abspath(
            f"{output_dir}/po_{file_path.split('/')[-2]}_{file_path.split('/')[-1].removesuffix('.csv')}",
        )
        for file_path in file_paths
    ]
    for file_path, new_file_path in zip(file_paths, new_file_paths):
        timestamps, x, u = load_old_data(file_path)
        last_delta = np.insert(u[:-1, 1], 0, 0)
        ddelta = u[:, 1] - last_delta
        T = u[:, 0]
        to_write = np.concatenate(
            (
                timestamps.reshape(-1, 1),
                x[:, :6],
                last_delta.reshape(-1, 1),
                np.zeros((x.shape[0], 4 * 40)),
                T.reshape(-1, 1),
                ddelta.reshape(-1, 1),
            ),
            axis=1,
        )
        np.savetxt(new_file_path, to_write, header=header, delimiter=",", fmt="%.15g")

    # now create two directories train and test and fill them with 90% and 10% of the data respectively
    # (move the files, don't copy them)
    train_dir = f"{output_dir}/train"
    test_dir = f"{output_dir}/test"
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    train_idx = np.random.choice(
        len(new_file_paths), int(0.9 * len(new_file_paths)), replace=False
    )
    test_idx = np.setdiff1d(np.arange(len(new_file_paths)), train_idx)
    for id in train_idx:
        shutil.move(new_file_paths[id], train_dir)
    for id in test_idx:
        shutil.move(new_file_paths[id], test_dir)


if __name__ == "__main__":
    # dataset_velocity_distribution()
    # portesouvertes_dataset_velocity_distribution()
    preprocess_data_portes_ouvertes()
