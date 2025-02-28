# cnn model functions 
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import tqdm

# create a class for the model
def createNet(printtoggle=False, lr=0.001, weight_decay=0.001, dropout=0.1):
    class visNet(nn.Module):
        def __init__(self, printtoggle):
            super().__init__()

            ### Convolutional layers (Reduced Complexity)
            # Input: 1 channel, Output: 10 feature maps, Kernel: 5x5, Padding: 1, Stride: 1
            self.conv1 = nn.Conv2d(1, 10, kernel_size=5, stride=1, padding=1)
            self.bnorm1 = nn.BatchNorm2d(10)

            # Input: 10 channels, Output: 20 feature maps, Kernel: 5x5, Padding: 1, Stride: 1
            self.conv2 = nn.Conv2d(10, 20, kernel_size=5, stride=1, padding=1)
            self.bnorm2 = nn.BatchNorm2d(20)

            # Input: 20 channels, Output: 30 feature maps, Kernel: 5x5, Padding: 1, Stride: 1
            self.conv3 = nn.Conv2d(20, 30, kernel_size=5, stride=1, padding=1)
            self.bnorm3 = nn.BatchNorm2d(30)

            # Let's assume the input is 76x4500
            # conv1 output size: (76-5+2*1)/1 + 1 = 74. pool: 74/2 = 37
            # conv2 output size: (37-5+2*1)/1 + 1 = 35. pool: 35/2 = 17 (rounded down)
            # conv3 output size: (17-5+2*1)/1 + 1 = 15. pool: 15/2 = 7 (rounded down)

            # similarly for the height:
            # conv1 output size: (4500-5+2*1)/1 + 1 = 4498. pool: 4498/2 = 2249
            # conv2 output size: (2249-5+2*1)/1 + 1 = 2247. pool: 2247/2 = 1123
            # conv3 output size: (1123-5+2*1)/1 + 1 = 1121. pool: 1121/2 = 560

            # Output size after conv3 and pool3: 7x560
            expectSize = 30 * 7 * 560
            
            ### Fully connected layers
            self.fc1 = nn.Linear(expectSize, 50)  # Reduced units in FC1
            self.out = nn.Linear(50, 5)  # Output layer (5 classes)

            # Reduced Dropout for regularization
            self.dropout = nn.Dropout(dropout) #added a dropout in the forward func

            self.print = printtoggle

        def forward(self, x):
            if self.print:
                print(f'Input: {x.shape}')

            # Conv1 -> BatchNorm -> ReLU -> MaxPool
            conv1act = F.relu(self.conv1(x))
            x = F.relu(self.bnorm1(F.max_pool2d(self.conv1(x),2)))
            if self.print:
                print(f'Layer conv1/pool1: {x.shape}')

            # Conv2 -> BatchNorm -> ReLU -> MaxPool
            conv2act = F.relu(self.conv2(x))
            x = F.relu(self.bnorm2(F.max_pool2d(self.conv2(x),2)))
            if self.print:
                print(f'Layer conv2/pool2: {x.shape}')

            # Conv3 -> BatchNorm -> ReLU -> MaxPool
            conv3act = F.relu(self.conv3(x))
            x = F.relu(self.bnorm3(F.max_pool2d(self.conv3(x),2)))
            if self.print:
                print(f'Layer conv3/pool3: {x.shape}')

            # Flatten for fully connected layers
            x = torch.flatten(x, 1)  # Flatten all dimensions except the batch
            if self.print:
                print(f'Vectorize: {x.shape}')

            # FC1 -> ReLU -> Dropout
            x = F.relu(self.fc1(x))
            x = self.dropout(x)  # Apply dropout for regularization
            if self.print:
                print(f'Layer fc1: {x.shape}')


            # Output layer
            x = self.out(x)
            if self.print:
                print(f'Layer out: {x.shape}')

            return x, conv1act, conv2act, conv3act

    net = visNet(printtoggle)
    lossfun = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)

    return net, lossfun, optimizer


# a function that trains the model
def funtion2trainTheModel(net, lossfun, optimizer, train_loader, test_loader, numepochs=30, device = 'cpu'):

  # send the model to the GPU
  net.to(device)

  # initialize losses
  trainLoss = torch.zeros(numepochs)
  devLoss   = torch.zeros(numepochs)
  trainAcc  = torch.zeros(numepochs)
  devAcc    = torch.zeros(numepochs)

  # loop over epochs
  for epochi in tqdm.tqdm(range(numepochs)):

    # loop over training data batches
    net.train() # switch to train mode
    batchLoss = []
    batchAcc  = []
    for X,y in train_loader:

      # push data to GPU
      X = X.to(device)
      y = y.to(device)

      # forward pass and loss
      yHat = net(X)[0]
      loss = lossfun(yHat,y)

      # backprop
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      # loss and accuracy from this batch
      batchLoss.append(loss.item())
      batchAcc.append( torch.mean((torch.argmax(yHat,axis=1) == y).float()).item() )
    
    # and get average losses and accuracies across the batches
    trainLoss[epochi] = np.mean(batchLoss)
    trainAcc[epochi]  = 100*np.mean(batchAcc)

    #### test performance
    net.eval() # switch to test mode
    X,y = next(iter(test_loader))

    # push data to GPU
    X = X.to(device)
    y = y.to(device)

    # forward pass and loss
    with torch.no_grad():
      yHat = net(X)[0]
      loss = lossfun(yHat,y)

    # and get average losses and accuracies across the batches
    devLoss[epochi] = loss.item()
    devAcc[epochi]  = 100*torch.mean((torch.argmax(yHat,axis=1) == y).float()).item()

  # end epochs

  # function output
  return trainLoss,devLoss,trainAcc,devAcc,net

def getDataLoaders(Data,labels, test_size = 0.2, batchsize = 32,  num_stim = 4500):
    # restrict to first 4500 stimuli
    Data = Data[:,:,:,:num_stim]

    # convert to tensor
    dataT   = torch.tensor( Data ).float()
    labelsT = torch.tensor( labels ).long()

    # use scikitlearn to split the data
    train_data,test_data, train_labels,test_labels = train_test_split(dataT, labelsT, test_size=test_size)

    # convert into PyTorch Datasets
    train_data = torch.utils.data.TensorDataset(train_data,train_labels)
    test_data  = torch.utils.data.TensorDataset(test_data,test_labels)

    # translate into dataloader objects
    batchsize    = 32
    train_loader = DataLoader(train_data,batch_size=batchsize,shuffle=True,drop_last=True)
    test_loader  = DataLoader(test_data,batch_size=test_data.tensors[0].shape[0])

    return train_loader,test_loader