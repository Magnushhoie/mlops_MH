# -*- coding: utf-8 -*-
import logging
import os
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf
from torch import nn

from src.models import tools
from src.models.CNN import CNN

CONF_PATH = Path(os.getcwd(), "config")


@hydra.main(config_path=CONF_PATH, config_name="main.yaml")
def evaluate(config):
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    os.chdir(hydra.utils.get_original_cwd())  # Avoid breaking relative path
    log.info(f"configuration: \n {OmegaConf.to_yaml(config)}")
    params = config
    hparams = config.experiment

    # verbose = params.verbose
    np.random.seed(params.seed)

    # Load model
    model = CNN()
    model = tools.load_checkpoint(model, params.model_path)
    criterion = nn.CrossEntropyLoss()

    # Load data
    log.info(f"\nLoading data from {params.dataset_path}")
    _, _, testloader = tools.load_train_valid_test(
        params.dataset_path, hparams.batch_size, v=False
    )
    log.info(f"Test set: {testloader.dataset.tensors[0].shape}")

    mean_loss, mean_accuracy = predict(model, testloader, criterion)
    print(mean_loss)
    return mean_loss


def predict(model, testloader, criterion):

    model.eval()
    testset = testloader.dataset.tensors[0]

    # loss_list = []
    # accuracy_list = []

    # Predict
    correct = 0
    running_loss = 0
    for i, (images, labels) in enumerate(testloader):

        output = model(images)
        loss = criterion(output, labels)

        # Loss and accuracy
        running_loss += loss.item()
        # ps = torch.exp(model(images))
        predicted = torch.max(output.data, 1)[1]
        correct += (predicted == labels).sum()

    mean_loss = running_loss / len(testset)
    mean_accuracy = (100 * correct / len(testset)).item()
    log.info(f"Test Loss: {mean_loss:.6f}, Accuracy: {mean_accuracy:.2f}")

    return mean_loss, mean_accuracy


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[{asctime}] {message}", style="{")
    log = logging.getLogger(__name__)
    log.info("Evaluating on test set")

    evaluate()
