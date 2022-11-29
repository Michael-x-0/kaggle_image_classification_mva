import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet152 as res, ResNet152_Weights as res_weight
import torchvision.transforms as transforms

weights = res_weight.IMAGENET1K_V2
preprocess = weights.transforms()


class Net(nn.Module):
  def __init__(self,n_classes, only_fc_layer, auto_path = None):
    super(Net, self).__init__()
    self.resnet = res(weights = res_weight.IMAGENET1K_V2)
    self.auto_path = auto_path
    if only_fc_layer:
      for param in self.resnet.parameters():
        param.requires_grad = False
    autoEncoder = AutoEncoder()
    if auto_path is not None:
      print('Loading auto encodeur model')
      state_dict = torch.load(auto_path)
      autoEncoder.load_state_dict(state_dict)
    self.resnet.fc = nn.Linear(2048,n_classes)
    self.autoEncoder = autoEncoder
    self.Linear1 = nn.Linear(200,n_classes)
    self.Linear2 = nn.Linear(n_classes + 5,n_classes)

  def forward(self,x):
    # out = self.resnet(x)
    # if self.auto_path is not None:
    #   _,x = self.autoEncoder(x)
    #   x = F.relu(self.Linear1(x.view(-1,588)))
    #   out = F.relu(out)
    #   out = self.Linear2(torch.cat((out,x),axis = -1))
    # return out 
    _,x = self.autoEncoder(x)
    return self.Linear1(x.view(-1,200))

class AutoEncoder(nn.Module):
  def __init__(self):
      super(AutoEncoder, self).__init__()
      self.conv1 = nn.Conv2d(3, 1, 3, padding=1)  
      self.pool = nn.MaxPool2d(4, 4)
      self.Linear1 = nn.Linear(3136,200)
      self.Linear2 = nn.Linear(200,3136)
      self.t_conv1 = nn.ConvTranspose2d(1, 3, 4, stride=4)

  def forward(self, x):
      x = F.relu(self.conv1(x))
      x = self.pool(x)
      x_enc = F.relu(self.Linear1(x.view(-1,3136)))
      x = F.relu(self.Linear2(x_enc))
      x = self.t_conv1(x.view(-1,1,56,56))
      return x,x_enc

