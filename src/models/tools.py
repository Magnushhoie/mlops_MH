import torch

def save_checkpoint(model, filepath="models/checkpoint.pth", v=False):
    print(f"Saving model to {filepath}")

    checkpoint = {"model": model,
                  "state_dict": model.state_dict()}
    torch.save(checkpoint, filepath)

    if v:
        print("Model: \n\n", model, '\n')
        print("The state dict keys: \n\n", model.state_dict().keys())

def load_checkpoint(filepath, v=True):
    print("Loading model ...")

    checkpoint = torch.load(filepath)

    model = checkpoint["model"]
    model.load_state_dict(checkpoint["state_dict"])

    if v:
        print(f"Model: \n\n {model}\n")
        print(f"The state dict keys: \n\n{model.state_dict().keys()}")
    
    return model

def test_model(model, trainloader, criterion, optimizer, v=True):
    dataiter = iter(trainloader)
    images, labels = next(dataiter)

    # Forward pass, get our logits
    logits = model(images)

    # Calculate the loss with the logits and the labels
    try:
        loss = criterion(logits, labels)
        if v: print(f"Success! Loss: {loss}")
    except Error as e:
        print(f"Error {e}")
        exit()