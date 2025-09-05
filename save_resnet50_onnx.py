from torchvision import models, datasets, transforms as T
import torch
from PIL import Image
import numpy as np

resnet50 = models.resnet50(pretrained=True)

# Export the model to ONNX
image_height = 224
image_width = 224
x = torch.randn(1, 3, image_height, image_width, requires_grad=True)
torch_out = resnet50(x)
torch.onnx.export(resnet50,                     # model being run
                  x,                            # model input (or a tuple for multiple inputs)
                  "src/resnet50.onnx",          # where to save the model (can be a file or file-like object)
                  export_params=True,           # store the trained parameter weights inside the model file
                  opset_version=16,             # the ONNX version to export the model to (Rust Burn requires ≥16)
                  do_constant_folding=True,     # whether to execute constant folding for optimization
                  input_names = ['input'],      # the model's input names
                  output_names = ['output'])    # the model's output names

