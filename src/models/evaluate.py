# -*- coding: utf-8 -*-
import glob
import logging
import os
from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from dotenv import find_dotenv, load_dotenv
from torch import nn
from torch.autograd import Variable

from src.models import tools
from src.models.CNN import CNN
from src.visualization import visualize

seed = 42
np.random.seed(seed)

# Params
verbose = True
epochs = 4
batch_size = 64


def predict(model, testloader, criterion):

    model.eval()
    testset = testloader.dataset.tensors[0]

    loss_list = []
    accuracy_list = []

    # Predict
    correct = 0
    running_loss = 0
    for i, (images, labels) in enumerate(testloader):

        output = model(images)
        loss = criterion(output, labels)

        # Loss and accuracy
        running_loss += loss.item()
        ps = torch.exp(model(images))
        predicted = torch.max(output.data, 1)[1]
        correct += (predicted == labels).sum()

    mean_loss = running_loss / len(testset)
    mean_accuracy = (100 * correct / len(testset)).item()
    print(f"Test Loss: {mean_loss:.6f}, Accuracy: {mean_accuracy:.2f}")

    return mean_loss, mean_accuracy


@click.command()
@click.argument(
    "model_filepath", default="models/checkpoint.pth", type=click.Path(exists=True)
)
@click.argument(
    "input_filepath", default="data/processed", type=click.Path(exists=True)
)
@click.argument("output_filepath", default="models", type=click.Path())
def main(model_filepath, input_filepath, output_filepath):
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info("making final data set from raw data")

    # Load model
    model = CNN()
    model = tools.load_checkpoint(model, model_filepath)
    criterion = nn.CrossEntropyLoss()

    # Load data
    print(f"\nLoading data from {input_filepath}")
    _, _, testloader = tools.load_train_valid_test(input_filepath, batch_size, v=False)
    print(f"Test set: {testloader.dataset.tensors[0].shape}")

    predict(model, testloader, criterion)

    print("\nDone!")


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
