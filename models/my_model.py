import torch
import torch.nn as nn
from torch.nn.modules import conv


class MyModel(nn.Module):
    # You can use pre-existing models but change layers to recieve full credit.
    def __init__(self):
        super(MyModel, self).__init__()
        #############################################################################
        # TODO: Initialize the network weights                                      #
        #############################################################################

        self.kernel = 3
        self.padding = 1
        self.output_channels1 = 32
        self.output_channels2 = 64
        self.output_channels3 = 128
        self.channels = 3

        self.kernel_pool = 2
        self.stride_pool = 2 
        
        self.output_features = 10
        
        
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2,2)
        self.lin1 = nn.Linear(64*16*16, 256)
        self.lin2 = nn.Linear(256, 10)
        

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

    def forward(self, x):
        outs = None
        #############################################################################
        # TODO: Implement forward pass of the network                               #
        #############################################################################
        
        N = x.shape[0]
  

        conv1 = self.conv1(x)
        conv1 = self.relu(conv1)
        conv2 = self.conv2(conv1)
        conv2 = self.relu(conv2)
        pool1 = self.pool(conv2)
      

        lin1 = self.lin1(pool1.view(N, -1))
        lin1 = self.relu(lin1)
        outs = self.lin2(lin1)
      
      

        


        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        return outs