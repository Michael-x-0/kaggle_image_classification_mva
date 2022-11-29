import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet152 as res, ResNet152_Weights as res_weight
import torchvision.transforms as transforms

weights = res_weight.IMAGENET1K_V2
preprocess = weights.transforms()

transform_autoEncodeur = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
])
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
      self.Linear = nn.Linear(n_classes + 400,n_classes)
    self.resnet.fc = nn.Linear(2048,n_classes)

  def forward(self,x):
    out = self.resnet(x)
    if self.autoEncoder is not None:
      x = transform_autoEncodeur(x)
      x,_,_ = self.autoEncodeur(x)
      out = F.Relu(self.Linear(torch.cat((out,x),axis = -1)))
    return out 

class AutoEncoder(nn.Module):
   def __init__(self, image_size=12288 , h_dim=1000, z_dim=200):
        super(AutoEncoder, self).__init__()
        self.fc1 = nn.Linear(image_size, h_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)
        self.fc3 = nn.Linear(h_dim, z_dim)
        self.fc4 = nn.Linear(z_dim, h_dim)
        self.fc5 = nn.Linear(h_dim, image_size)
        
   def encode(self, x):
      h = F.relu(self.fc1(x))
      return self.fc2(h), self.fc3(h)
  
   def reparameterize(self, mu, log_var):
      std = torch.exp(log_var/2)
      eps = torch.randn_like(std)
      return mu + eps * std

   def decode(self, z):
      h = F.relu(self.fc4(z))
      return torch.sigmoid(self.fc5(h))
  
   def forward(self, x):
      mu, log_var = self.encode(x)
      z = self.reparameterize(mu, log_var)
      x_reconst = self.decode(z)
      return x_reconst, mu, log_var

