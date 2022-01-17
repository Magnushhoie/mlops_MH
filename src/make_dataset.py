# -*- coding: utf-8 -*-
import glob
import logging
import os
from pathlib import Path

import hydra
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset

CONF_PATH = Path(os.getcwd(), "config")


@hydra.main(config_path=CONF_PATH, config_name="main.yaml")
def make_dataset(config):
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    os.chdir(hydra.utils.get_original_cwd())  # Avoid breaking relative path
    params = config

    if params.all_train_partitions == "False":
        all_train_partitions_bool = False
    else:
        all_train_partitions_bool = True

    # Extract and save data
    generate_train_valid_test(
        dataDir=params.raw_path,
        outDir=params.dataset_path,
        all_train_partitions_bool=all_train_partitions_bool,
        subset_fraction=params.valid_fraction,
        seed=params.seed,
        verbose=params.verbose,
    )


def generate_train_valid_test(
    dataDir, outDir, all_train_partitions_bool=False, subset_fraction=0.10, seed=42, verbose=True
):
    """Saves pre-processed tensor datasets from loaded files in dataDir

    Args:
        dataDir: Directory with training and testing numpy array files
        outDir: Directory to save pre-processed train, validation and test dataset tensor files
        subset: If True, Load only one training file
        subset_fraction: Fraction of training dataset to use for validation
        seed: Seed used for training / validation split

    Returns:
        Nothing
    """
    train_list = glob.glob(dataDir + "/train*.npz")
    test_list = glob.glob(dataDir + "/test*.npz")

    def load_data(file_list, v=True):
        """Loads numpy array data such as training set from list of files"""
        if v:
            log.info(f"Loading data from: {file_list}")

        images_list, labels_list = [], []
        for i in range(len(file_list)):
            images, labels = (
                np.load(file_list[i])["images"],
                np.load(file_list[i])["labels"],
            )
            images_list.append(images)
            labels_list.append(labels)

        # Stack to N x 28 x 28
        Images, Labels = np.vstack(images_list), np.concatenate(labels_list)

        # Reshape to N x 1 x 28 x 28
        Images = np.expand_dims(Images, axis=1)

        # Correct dtypes
        Images = Images.astype(np.float32)
        Labels = Labels.astype(np.int64)

        return (Images, Labels)

    if not all_train_partitions_bool:
        train_list = train_list[0:1]

    # Training and validation dataset
    X_train, y_train = load_data(train_list)

    # Split by subset_fraction
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train, y_train, test_size=subset_fraction, random_state=seed
    )

    # Test dataset
    X_test, y_test = load_data(test_list)

    if verbose:
        log.info(f"Subsampled {subset_fraction} of training dataset ...")
        log.info(f"X_train {X_train.shape}, y_train {y_train.shape}")
        log.info(f"X_valid {X_valid.shape}, y_valid {y_valid.shape}")
        log.info(f"X_test {X_test.shape}, y_test {y_test.shape}")

    # Convert to tensor
    X_train, y_train = torch.from_numpy(X_train), torch.from_numpy(y_train)
    X_valid, y_valid = torch.from_numpy(X_valid), torch.from_numpy(y_valid)
    X_test, y_test = torch.from_numpy(X_test), torch.from_numpy(y_test)

    # Save as torch dataset
    if verbose:
        log.info(f"Saving train.pt, valid.pt and test.pt to {outDir}")

    torch.save(TensorDataset(X_train, y_train), outDir + "/train.pt")
    torch.save(TensorDataset(X_valid, y_valid), outDir + "/valid.pt")
    torch.save(TensorDataset(X_test, y_test), outDir + "/test.pt")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[{asctime}] {message}", style="{")
    log = logging.getLogger(__name__)
    log.info("making dataset")

    make_dataset()
