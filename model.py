import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet152 as res, ResNet152_Weights as res_weight

weights = res_weight.IMAGENET1K_V2
preprocess = weights.transforms()

class Net(nn.Module):
  def __init__(self,n_classes, only_fc_layer):
    super(Net, self).__init__()
    self.resnet = res(weights = res_weight.IMAGENET1K_V2)
    if only_fc_layer:
      for param in self.resnet.parameters():
        param.requires_grad = False
    self.resnet.fc = nn.Linear(2048,n_classes)

  def forward(self,x):
    return self.resnet(x)
