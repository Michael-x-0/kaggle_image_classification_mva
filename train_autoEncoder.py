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


from data import data_transforms

dataset = datasets.ImageFolder(args.data,
                         transform=data_transforms)
train_loader = torch.utils.data.DataLoader(dataset
    ,
    batch_size=args.batch_size, shuffle=True, num_workers=1)

# number of epochs to train the model
n_epochs = args.epochs

criterion = nn.BCELoss()
# specify loss function
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)

for epoch in range(1, n_epochs+1):
    # monitor training loss
    train_loss = 0.0
    
    ###################
    # train the model #
    ###################
    for batch_idx, (data, _) in enumerate(train_loader):
        # _ stands in for labels, here
        # no need to flatten images
        # clear the gradients of all optimized variables
        if use_cuda:
            data = data.cuda()
        optimizer.zero_grad()
        # forward pass: compute predicted outputs by passing inputs to the model
        outputs, x_enc = model(data)
        # calculate the loss
        loss = criterion(outputs, data)
        # backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()
        # perform a single optimization step (parameter update)
        optimizer.step()
        # update running training loss
        train_loss += loss.item()*data.size(0)
        
            
    # print avg training statistics 
    train_loss = train_loss/len(train_loader)
    print('Epoch: {} \tTraining Loss: {:.6f}'.format(
        epoch, 
        train_loss
        ))
model_file = args.experiment + '/model_autoencoder''_epoch_' + str(epoch) + '.pth'
torch.save(model.state_dict(), model_file)
print('model saved at {}'.format(model_file))




