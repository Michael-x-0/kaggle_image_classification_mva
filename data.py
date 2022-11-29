import zipfile
import os

import torchvision.transforms as transforms
from torchvision.models import ResNet50_Weights
# once the images are loaded, how do we pre-process them before being passed into the network
# by default, we resize the images to 64 x 64 in size
# and normalize them to mean = 0 and standard-deviation = 1 based on statistics collected from
# the training set

from model import preprocess


data_augmentation = transforms.Compose(
  [transforms.Resize((232,232)),
  transforms.RandomCrop(224),
  transforms.RandomHorizontalFlip(0.5),
  transforms.RandomVerticalFlip(0.5),
  transforms.RandomRotation((0,90)),
  transforms.GaussianBlur(3,(0.1,2)),
  preprocess
  ]
)
data_transforms = preprocess

