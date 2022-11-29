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
    if auto_path is not None:
      state_dict = torch.load(auto_path)
      autoEncoder = AutoEncoder()
      autoEncoder.load_state_dict(state_dict)
      self.autoEncoder = autoEncoder
      self.Linear = nn.Linear(n_classes + 588,n_classes)
    self.resnet.fc = nn.Linear(2048,n_classes)

  def forward(self,x):
    out = self.resnet(x)
    if self.autoEncoder is not None:
      _,x = self.autoEncodeur(x)
      out = F.Relu(self.Linear(torch.cat((out,x.view(-1)),axis = -1)))
    return out 

class AutoEncoder(nn.Module):
  def __init__(self):
      super(AutoEncoder, self).__init__()
      ## encoder layers ##
      # conv layer (depth from 3 --> 16), 3x3 kernels
      self.conv1 = nn.Conv2d(3, 16, 3, padding=1)  
      self.conv2 = nn.Conv2d(16, 10, 3, padding=1)
      self.conv3 = nn.Conv2d(10, 5, 3, padding=1)
      self.conv4 = nn.Conv2d(5, 3, 3, padding=1)
      # pooling layer to reduce x-y dims by two; kernel and stride of 2
      self.pool = nn.MaxPool2d(2, 2)
      
      ## decoder layers ##
      ## a kernel of 2 and a stride of 2 will increase the spatial dims by 2
      self.t_conv1 = nn.ConvTranspose2d(3, 5, 2, stride=2)
      self.t_conv2 = nn.ConvTranspose2d(5, 10, 2, stride=2)
      self.t_conv3 = nn.ConvTranspose2d(10, 16, 2, stride=2)
      self.t_conv4 = nn.ConvTranspose2d(16, 3, 2, stride=2)

  def forward(self, x):
      ## encode ##
      # add hidden layers with relu activation function
      # and maxpooling after
      x = F.relu(self.conv1(x))
      x = self.pool(x)
      x = F.relu(self.conv2(x))
      x = self.pool(x)
      x = F.relu(self.conv3(x))
      x = self.pool(x)
      x = F.relu(self.conv4(x))
      x_enc = self.pool(x)
      
      ## decode ##
      # add transpose conv layers, with relu activation function
      x = F.relu(self.t_conv1(x_enc))
      x = F.relu(self.t_conv2(x))
      x = F.relu(self.t_conv3(x))
      x = self.t_conv4(x)
              
      return x,x_enc

