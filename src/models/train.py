# -*- coding: utf-8 -*-
import logging
from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np
import torch
from dotenv import find_dotenv, load_dotenv
from torch import nn

from src.models import tools
from src.models.CNN import CNN
from src.visualization import visualize

seed = 42
np.random.seed(seed)

# Params
verbose = True
epochs = 4
batch_size = 64


def fit(model, epochs, trainloader, criterion, optimizer):

    model.train()
    trainset = trainloader.dataset.tensors[0]

    loss_list = []
    accuracy_list = []

    for epoch in range(epochs):
        print(f"Epoch {epoch+1} / {epochs}")

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
        print(f"Training Loss: {mean_loss:.6f}, Accuracy: {mean_accuracy:.2f}")

        # Stats
        loss_list.append(mean_loss)
        accuracy_list.append(mean_accuracy)

    return loss_list, accuracy_list


@click.command()
@click.argument(
    "input_filepath", default="data/processed/", type=click.Path(exists=True)
)
@click.argument("output_filepath", default="models/", type=click.Path())
def main(input_filepath, output_filepath):
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info("making final data set from raw data")

    # Extract and save data
    model = CNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())  # ,lr=0.001, betas=(0.9,0.999))

    if verbose:
        print("Criterion", criterion)
        print("Optimizer", optimizer)
        print("Model: \n\n", model, "\n")
        print("The state dict keys: \n\n", model.state_dict().keys())

    # Load data
    print(f"\nLoading data from {input_filepath}")
    trainloader, validloader, testloader = tools.load_train_valid_test(
        input_filepath, batch_size, v=verbose
    )

    # Test
    tools.test_model(model, trainloader, criterion, optimizer, v=False)

    # Train
    print("\nTraining model ...")
    epochs = 2
    loss_list, accuracy_list = fit(model, epochs, trainloader, criterion, optimizer)

    # Save
    tools.save_checkpoint(model)

    # Visualize
    print(f"\nVisualizing data, saving to {output_filepath}")

    visualize.plot_metric(loss_list, "Loss")
    filepath = output_filepath + "/train_loss.pdf"
    print(f"Saving {filepath}")
    plt.savefig(filepath)

    visualize.plot_metric(accuracy_list, "Accuracy")
    filepath = output_filepath + "/train_accuracy.pdf"
    print(f"Saving {filepath}")
    plt.savefig(filepath)

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
