
import torch
import torchvision.models as models
from torch.profiler import profile, record_function, ProfilerActivity
from torch.profiler import profile, tensorboard_trace_handler

model = models.resnet18()
inputs = torch.randn(5, 3, 224, 224)

with profile(activities=[ProfilerActivity.CPU],
            record_shapes=True,
            on_trace_ready=tensorboard_trace_handler("profiler")) as prof:
   with record_function("model_inference"):
      model(inputs)
