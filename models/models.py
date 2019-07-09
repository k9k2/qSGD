from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class Linear(nn.Module):
    def __init__(self, dx, num_classes):
        super(Linear, self).__init__()
        self.dx = dx
        self.fc = nn.Linear(self.dx, num_classes)
    def forward(self, x):
        out = self.fc(x.view(-1,self.dx))
        return out

class LeNet(nn.Module):
    def __init__(self, nc, nh, hw, num_classes):
        input_shape = (nc,nh,hw)
        super(LeNet, self).__init__()
        self.maxpool = nn.MaxPool2d((2,2))      
        self.conv1 = nn.Conv2d(nc,64,5)    
        self.conv2 = nn.Conv2d(64,64,5)        
        self.flat_shape = self.get_flat_shape(input_shape)            
        self.fc1 = nn.Linear(self.flat_shape, 1024)
        self.fc2 = nn.Linear(1024, num_classes)

    def get_flat_shape(self, input_shape):
        dummy = Variable(torch.zeros(1, *input_shape))
        dummy = self.maxpool(self.conv1(dummy))
        dummy = self.maxpool(self.conv2(dummy))
        return dummy.data.view(1, -1).size(1)
        
    def forward(self, x_in):        
        # conv 1
        x = self.conv1(x_in)
        x = self.maxpool(x)
        x = F.relu(x)
        # conv 2
        x = self.conv2(x)
        x = self.maxpool(x)
        x = F.relu(x)
        # flatten
        x = x.view(-1,self.flat_shape)
        # fc 1
        x = self.fc1(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        # fc 2
        x_out1 = self.fc2(x)    
        return x_out1      

