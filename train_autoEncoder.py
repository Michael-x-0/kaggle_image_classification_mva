import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets
from torch.autograd import Variable
from tqdm import tqdm
import torch.nn.functional as F

parser = argparse.ArgumentParser(description='Training Auto Encodeur')
parser.add_argument('--tensorboard_log_dir', type = str, help = 'path for tensorboard output', required=True)
parser.add_argument('--data', type=str, default='bird_dataset', metavar='D',
                    help="folder where data is located. train_images/ and val_images/ need to be found in the folder")
parser.add_argument('--batch-size', type=int, default=64, metavar='B',
                    help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--experiment', type=str, default='experiment', metavar='E',
                    help='folder where experiment outputs are located.')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
args = parser.parse_args()
use_cuda = torch.cuda.is_available()
torch.manual_seed(args.seed)

# Create experiment folder
if not os.path.isdir(args.experiment):
    os.makedirs(args.experiment)

from model import AutoEncoder
model = AutoEncoder()
if use_cuda:
    print('Using GPU')
    model.cuda()
else:
    print('Using CPU')


optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

from data import data_transforms
dataset1 = datasets.ImageFolder(args.data + '/train_images',
                         transform=data_transforms)
dataset2 = datasets.ImageFolder(args.data + '/val_images',
                         transform=data_transforms)
dataset3 = datasets.ImageFolder(args.data + '/test_images',
                         transform=data_transforms)
f_dataset = torch.utils.data.ConcatDataset([dataset1,dataset2, dataset3])
train_loader = torch.utils.data.DataLoader( f_dataset
    ,
    batch_size=args.batch_size, shuffle=True, num_workers=1)
for epoch in range(args.epochs):
    for i, (x, _) in enumerate(data_loader):
        # Forward pass
        x = x.to(device).view(-1, image_size)
        x_reconst, mu, log_var = model(x)
        
        # Compute reconstruction loss and kl divergence
        # For KL divergence between Gaussians, see Appendix B in VAE paper or (Doersch, 2016):
        # https://arxiv.org/abs/1606.05908
        reconst_loss = F.binary_cross_entropy(x_reconst, x, reduction='sum')
        kl_div = - 0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        
        # Backprop and optimize
        loss = reconst_loss + kl_div
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1) % 10 == 0:
            print ("Epoch[{}/{}], Step [{}/{}], Reconst Loss: {:.4f}, KL Div: {:.4f}" 
                   .format(epoch+1, num_epochs, i+1, len(data_loader), reconst_loss.item()/batch_size, kl_div.item()/batch_size))

model_file = args.experiment + '/model_autoencoder''_epoch_' + str(epoch) + '.pth'
torch.save(model.state_dict(), model_file)






