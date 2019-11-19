"""
This module implements a multi-layer perceptron (MLP) in NumPy.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from modules import *

class MLP(object):
  """
  This class implements a Multi-layer Perceptron in NumPy.
  It handles the different layers and parameters of the model.
  Once initialized an MLP object can perform forward and backward.
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
    hidden = []
    n_in = n_inputs
    n_out = n_hidden[0]
    for it in range(len(n_hidden)):
      hidden.append(modules.LinearModule(n_in, n_out))
      hidden.append(modules.LeakyReLUModule(neg_slope))

      if not(it == len(n_hidden)-1):
        n_in = n_out
        n_out = n_hidden(it+1)

    self.softmax = modules.SoftMaxModule()

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
    #go through all hidden layers (incl relu)
    for it in range(len(self.hidden)):
      x = self.hidden[it](x)

    out = self.softmax(x)

    ########################
    # END OF YOUR CODE    #
    #######################

    return out

  def backward(self, dout):
    """
    Performs backward pass given the gradients of the loss. 

    Args:
      dout: gradients of the loss
    
    TODO:
    Implement backward pass of the network.
    """
    
    ########################
    # PUT YOUR CODE HERE  #
    #######################

    gradient = self.softmax.backward(dout)
    for it in range(len(self.hidden)):
      gradient = self.hidden[it].backward(gradient)

    #as the modules save the gradients for training we don't need to do that here again.
    ########################
    # END OF YOUR CODE    #
    #######################

    return
