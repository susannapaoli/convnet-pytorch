import torch
import torch.nn as nn

class TwoLayerNet(nn.Module):
    def __init__(self, input_dim, hidden_size, num_classes):
        '''
        :param input_dim: input feature dimension
        :param hidden_size: hidden dimension
        :param num_classes: total number of classes
        '''
        super(TwoLayerNet, self).__init__()
        #############################################################################
        # TODO: Initialize the TwoLayerNet, use sigmoid activation between layers   #
        #############################################################################
        self.lin1 = nn.Linear(input_dim, hidden_size)
        self.sigmoid = nn.Sigmoid()
        self.lin2 = nn.Linear(hidden_size, num_classes)
        self.softmax = nn.Softmax(dim=1)
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

    def forward(self, x):
        out = None
        #############################################################################
        # TODO: Implement forward pass of the network                               #
        #############################################################################
        N = x.shape[0]
        x = x.view(N, -1)
        lin1 = self.lin1(x)
        sigm = self.sigmoid(lin1)
        lin2 = self.lin2(sigm)
        out = self.softmax(lin2)
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        return out