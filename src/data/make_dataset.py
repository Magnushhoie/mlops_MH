# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

import numpy as np
import os, glob

from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import TensorDataset

seed = 42
np.random.seed(seed)

# Params
dataDir = "../data/raw/corruptmnist/"
outDir = "../data/processed/"
verbose = True
subset = True
subset_fraction = 0.10

def extract_save_train_valid_test(dataDir, outDir, subset=True, subset_fraction=0.10, seed=seed):
    """ Saves pre-processed tensor datasets from loaded files in dataDir
    
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
        """ Loads numpy array data such as training set from list of files """
        if v:
            print("Loading data from:", file_list, end="\n")

        images_list, labels_list = [], []
        for i in range(len(file_list)):
            images, labels = np.load(file_list[i])["images"], np.load(file_list[i])["labels"]
            images_list.append(images)
            labels_list.append(labels)

        # Stack to N x 28 x 28
        Images, Labels = np.vstack(images_list), np.concatenate(labels_list)

        # Reshape to N x 1 x 28 x 28
        Images = np.expand_dims(Images, axis=1)

        # Correct dtypes
        Images = Images.astype(np.float32)
        Labels = Labels.astype(np.int64)

        return(Images, Labels)

    if subset:
        train_list = train_list[0:1]

    # Training and validation dataset
    X_train, y_train = load_data(train_list)

    # Split by subset_fraction
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train,
                                         test_size=subset_fraction,
                                         random_state=seed)

    # Test dataset
    X_test, y_test = load_data(test_list)

    if verbose:
        print(f"Subsampled {subset_fraction} of training dataset ...")
        print(f"X_train {X_train.shape}, y_train {y_train.shape}")
        print(f"X_valid {X_valid.shape}, y_valid {y_valid.shape}")
        print(f"X_test {X_test.shape}, y_test {y_test.shape}")

    # Convert to tensor
    X_train, y_train = torch.from_numpy(X_train), torch.from_numpy(y_train)
    X_valid, y_valid = torch.from_numpy(X_valid), torch.from_numpy(y_valid)
    X_test, y_test = torch.from_numpy(X_test), torch.from_numpy(y_test)

    # Save as torch dataset
    if verbose:
        print(f"Saving train.pt, valid.pt and test.pt to {outDir}")

    torch.save(TensorDataset(X_train, y_train), outDir + "/train.pt")
    torch.save(TensorDataset(X_valid, y_valid), outDir + "/valid.pt")
    torch.save(TensorDataset(X_test, y_test), outDir + "/test.pt")

@click.command()
@click.argument('input_filepath', default="data/raw/corruptmnist", type=click.Path(exists=True))
@click.argument('output_filepath', default="data/processed/", type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    # Extract and save data
    extract_save_train_valid_test(dataDir=input_filepath, outDir=output_filepath, subset=True, subset_fraction=0.10, seed=seed)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
