from torchvision import models
import torch
model = models.resnet18()
script_model = torch.jit.script(model)
script_model.save("deployable_model.pt")
