# -*- coding: utf-8 -*-
import logging
import os
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf
from torch import nn

from src.models import tools
from src.models.CNN import CNN
from src.visualization import visualize

CONF_PATH = Path(os.getcwd(), "config")


@hydra.main(config_path=CONF_PATH, config_name="main.yaml")
def train(config):
    os.chdir(hydra.utils.get_original_cwd())  # Avoid breaking relative path
    log.info(f"configuration: \n {OmegaConf.to_yaml(config)}")
    params = config
    hparams = config.experiment

    verbose = params.verbose
    np.random.seed(hparams.seed)

    model = CNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=hparams.learning_rate
    )  # ,lr=0.001, betas=(0.9,0.999))

    # Load data
    log.info(f"\nLoading data from {params.dataset_path}")

    trainloader, validloader, testloader = tools.load_train_valid_test(
        params.dataset_path, hparams.batch_size, v=verbose
    )

    # Test
    tools.test_model(model, trainloader, criterion, optimizer, v=False)

    # Train
    log.info("\nTraining model ...")
    epochs = hparams.epochs
    loss_list, accuracy_list = fit(model, epochs, trainloader, criterion, optimizer)

    # Save
    tools.save_checkpoint(model)

    # Visualize
    log.info(f"\nVisualizing data, saving to {params.output_dir}")

    visualize.plot_metric(loss_list, "Loss")
    filepath = params.output_dir + "/train_loss.pdf"
    log.info(f"Saving {filepath}")
    plt.savefig(filepath)

    visualize.plot_metric(accuracy_list, "Accuracy")
    filepath = params.output_dir + "/train_accuracy.pdf"
    log.info(f"Saving {filepath}")
    plt.savefig(filepath)

    print(loss_list[-1])


def fit(model, epochs, trainloader, criterion, optimizer):
    model.train()
    trainset = trainloader.dataset.tensors[0]

    loss_list = []
    accuracy_list = []

    for epoch in range(epochs):
        log.info(f"Epoch {epoch+1} / {epochs}")

        correct = 0
        running_loss = 0
        for i, (images, labels) in enumerate(trainloader):

            optimizer.zero_grad()
            output = model(images)

            loss = criterion(output, labels)

            loss.backward()
            optimizer.step()

            # Loss and accuracy
            running_loss += loss.item()
            # ps = torch.exp(model(images))
            predicted = torch.max(output.data, 1)[1]
            correct += (predicted == labels).sum()

        mean_loss = running_loss / len(trainset)
        mean_accuracy = (100 * correct / len(trainset)).item()
        log.info(f"Training Loss: {mean_loss:.6f}, Accuracy: {mean_accuracy:.2f}")

        # Stats
        loss_list.append(mean_loss)
        accuracy_list.append(mean_accuracy)

    return loss_list, accuracy_list


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[{asctime}] {message}", style="{")
    log = logging.getLogger(__name__)
    log.info("fitting on train set")

    train()
