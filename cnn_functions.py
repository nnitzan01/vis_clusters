# cnn model functions 
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import tqdm

# create a class for the model
def createNet(printtoggle=False, lr = 0.001, weight_decay=0.001):
  class visNet(nn.Module):
    def __init__(self,printtoggle):
      super().__init__()

      ### convolution layers (input chan x feature maps)
      self.conv1 = nn.Conv2d(1,10,kernel_size=(5,5),stride=1,padding=1)
      self.bnorm1 = nn.BatchNorm2d(10) 
      # width  = np.floor((76 + 2*1 - 5)/1)+1 = 74/2 = 37
      # height = np.floor((4500 + 2*1 - 5)/1)+1 = 4498/2 = 2249

      self.conv2 = nn.Conv2d(10,20,kernel_size=(5,5),stride=1,padding=1)
      self.bnorm2 = nn.BatchNorm2d(20) 
      # width  = np.floor( (37+2*1-5)/1 )+1 = 36/2 = 17 
      # height = np.floor((2249 + 2*1 - 5)/1)+1 = 2247/2 = 1123

      self.conv3 = nn.Conv2d(20,30,kernel_size=(5,5),stride=1,padding=1)
      self.bnorm3 = nn.BatchNorm2d(30)  
      # width  = np.floor( (17+2*1-5)/1 )+1 = 15/2 = 7
      # height = np.floor((1123 + 2*1 - 5)/1)+1 = 1121/2 = 560

      # compute the number of units in FClayer (number of outputs of conv3)
      expectSize1 = np.floor( (7+2*0-1)/1 ) + 1
      expectSize2 = np.floor( (560+2*0-1)/1 ) + 1
      expectSize = 30*int(expectSize1*expectSize2)

      ### fully-connected layer
      self.fc1 = nn.Linear(expectSize,50)

      ### output layer
      self.out = nn.Linear(50,5)

      # toggle for printing out tensor sizes during forward prop
      self.print = printtoggle

    # forward pass
    def forward(self,x):

      print(f'Input: {x.shape}') if self.print else None

      # convolution -> maxpool -> relu
      # x = F.relu(F.max_pool2d(self.conv1(x),2))
      conv1act = F.relu(self.conv1(x))
      # x = F.avg_pool2d(conv1act,(2,2))
      x = F.relu(self.bnorm1(F.max_pool2d(self.conv1(x),(2,2))))
      print(f'Layer conv1/pool1: {x.shape}') if self.print else None

      # and again: convolution -> maxpool -> relu
      # x = F.relu(F.max_pool2d(self.conv2(x),2))
      conv2act = F.relu(self.conv2(x))
      # x = F.avg_pool2d(conv2act,(2,2))
      x = F.relu(self.bnorm2(F.max_pool2d(self.conv2(x),(2,2))))
      print(f'Layer conv2/pool2: {x.shape}') if self.print else None

      # and again: convolution -> maxpool -> relu
      # x = F.relu(F.max_pool2d(self.conv2(x),2))
      conv3act = F.relu(self.conv3(x))
      # x = F.avg_pool2d(conv3act,(2,2))
      x = F.relu(self.bnorm3(F.max_pool2d(self.conv3(x),(2,2))))
      print(f'Layer conv3/pool3: {x.shape}') if self.print else None

      # reshape for linear layer
      nUnits = x.shape.numel()/x.shape[0]
      x = x.view(-1,int(nUnits))
      if self.print: print(f'Vectorize: {x.shape}')

      # linear layers
      x = F.relu(self.fc1(x))
      if self.print: print(f'Layer fc1: {x.shape}')
      x = self.out(x)
      if self.print: print(f'Layer out: {x.shape}')

      return x, conv1act, conv2act, conv3act

  # create the model instance
  net = visNet(printtoggle)

  # loss function
  lossfun = nn.CrossEntropyLoss()

  # optimizer
  optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)

  return net,lossfun,optimizer


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

def getDataLoaders(Data,labels, batchsize = 32,  num_stim = 4500):
    # restrict to first 4500 stimuli
    Data = Data[:,:,:,:num_stim]

    # convert to tensor
    dataT   = torch.tensor( Data ).float()
    labelsT = torch.tensor( labels ).long()

    # use scikitlearn to split the data
    train_data,test_data, train_labels,test_labels = train_test_split(dataT, labelsT, test_size=.1)

    # convert into PyTorch Datasets
    train_data = torch.utils.data.TensorDataset(train_data,train_labels)
    test_data  = torch.utils.data.TensorDataset(test_data,test_labels)

    # translate into dataloader objects
    batchsize    = 32
    train_loader = DataLoader(train_data,batch_size=batchsize,shuffle=True,drop_last=True)
    test_loader  = DataLoader(test_data,batch_size=test_data.tensors[0].shape[0])

    return train_loader,test_loader