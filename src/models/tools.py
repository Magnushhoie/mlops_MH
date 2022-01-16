import torch

def load_train_valid_test(dataDir, batch_size, shuffle=True, v=True):
    # Load datasets
    trainset = torch.load(dataDir + "/train.pt")
    trainloader = torch.utils.data.DataLoader(trainset, batch_size, shuffle=True)

    validset = torch.load(dataDir + "/valid.pt")
    validloader = torch.utils.data.DataLoader(validset, batch_size, shuffle=True)

    testset = torch.load(dataDir + "/test.pt")
    testloader = torch.utils.data.DataLoader(testset, batch_size, shuffle=True)

    if v:
        print(f"Train: {trainloader.dataset.tensors[0].shape}")
        print(f"Valid: {validloader.dataset.tensors[0].shape}")
        print(f"Test: {testloader.dataset.tensors[0].shape}")

    return trainloader, validloader, testloader

def save_checkpoint(model, filepath="models/checkpoint.pth", v=False):
    print(f"Saving models.state_dict to {filepath}")
    torch.save(model.state_dict(), filepath)

    if v:
        print(f"Model:\n{model}")
        print(f"The state dict keys:\n{model.state_dict().keys()}")

def load_checkpoint(model, filepath="models/checkpoint.pth", v=True):
    print(f"Loading model.state_dict from {filepath}")
    model.load_state_dict(torch.load(filepath))

    if v:
        print(f"Model:\n{model}")
        print(f"The state dict keys:\n{model.state_dict().keys()}")
    
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