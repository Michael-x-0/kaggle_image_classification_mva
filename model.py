import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights

weights = ResNet50_Weights.IMAGENET1K_V2
preprocess = weights.transforms()

class Net(nn.Module):
  def __init__(self,n_classes, only_fc_layer):
    super(Net, self).__init__()
    self.resnet = resnet50(weights = ResNet50_Weights.IMAGENET1K_V2)
    if only_fc_layer:
      for param in self.resnet.parameters():
        param.requires_grad = False
    self.resnet.fc = nn.Linear(2048,n_classes)

  def forward(self,x):
    return self.resnet(x)
