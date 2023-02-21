import torch
import torch.nn as nn

class VanillaCNN(nn.Module):
    def __init__(self):
        super(VanillaCNN, self).__init__()
        #############################################################################
        # TODO: Initialize the Vanilla CNN                                          #
        #       Conv: 7x7 kernel, stride 1 and padding 0                            #
        #       Max Pooling: 2x2 kernel, stride 2                                   #
        #############################################################################
        self.kernel = 7
        self.padding = 0
        self.stride = 1
        self.kernel_pool = 2
        self.stride_pool = 2 
        self.output_channels = 32
        self.output_features = 10
        self.channels = 3

        self.conv = nn.Conv2d(self.channels, self.output_channels, kernel_size = self.kernel, stride = self.stride, padding = self.padding )
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size = self.kernel_pool, stride = self.stride_pool)

        W = (self.output_channels - self.kernel + 1) / ( 1+ self.stride)
        H = (self.output_channels - self.kernel + 1) / ( 1+ self.stride)

        self.lin = nn.Linear(int(self.output_channels * W * H), self.output_features)

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################


    def forward(self, x):
        outs = None
        #############################################################################
        # TODO: Implement forward pass of the network                               #
        #############################################################################
        N = x.shape[0]
        conv = self.conv(x)
        relu = self.relu(conv)
        out = self.maxpool(relu)
        out = out.view(N, -1)
        outs = self.lin(out)
        
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

        return outs