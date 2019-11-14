"""
This module implements a multi-layer perceptron (MLP) in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
  """
  This class implements a Multi-layer Perceptron in PyTorch.
  It handles the different layers and parameters of the model.
  Once initialized an MLP object can perform forward.
  """

  def __init__(self, n_inputs, n_hidden, n_classes, neg_slope):
    """
    Initializes MLP object. 
    
    Args:
      n_inputs: number of inputs.
      n_hidden: list of ints, specifies the number of units
                in each linear layer. If the list is empty, the MLP
                will not have any linear layers, and the model
                will simply perform a multinomial logistic regression.
      n_classes: number of classes of the classification problem.
                 This number is required in order to specify the
                 output dimensions of the MLP
      neg_slope: negative slope parameter for LeakyReLU

    TODO:
    Implement initialization of the network.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################

    #go through all the hidden layers
    n_in = []
    n_out = []
    for i in range(len(n_hidden) + 1):
      if i == 0:
        n_in.append(n_inputs)
      else:
        n_in.append(n_hidden[i-1])

      if i == len(n_hidden):
        n_out.append(n_classes)
      else:
        n_out.append(n_hidden[i])

    #fix error AttributeError: cannot assign module before Module.__init__() call
    super(MLP, self).__init__()
    #use nn internal sequential
    self.hidden = nn.ModuleList([nn.Linear(n_in[i], n_out[i]) for i in range(len(n_hidden)+1)])

    self.neg_slope = neg_slope
    ########################
    # END OF YOUR CODE    #
    #######################

  def forward(self, x):
    """
    Performs forward pass of the input. Here an input tensor x is transformed through 
    several layer transformations.
    
    Args:
      x: input to the network
    Returns:
      out: outputs of the network
    
    TODO:
    Implement forward pass of the network.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################

    #compute activity of all hidden units
    for i in range(len(self.hidden)-1):
      x = F.leaky_relu(self.hidden[i](x), self.neg_slope )
    #from last hidden to output layer
    x = self.hidden[-1](x)

    #softmax
    out = F.softmax(x, 1)
    ########################
    # END OF YOUR CODE    #
    #######################

    return out
