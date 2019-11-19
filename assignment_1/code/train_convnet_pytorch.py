"""
This module implements training and evaluation of a Convolutional Neural Network in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from convnet_pytorch import ConvNet
import cifar10_utils
import utils_custom
import torch
import torch.nn as nn
import torch.optim as optim

# Default constants
LEARNING_RATE_DEFAULT = 1e-4
BATCH_SIZE_DEFAULT = 32 #32
MAX_STEPS_DEFAULT = 5000 #5000
EVAL_FREQ_DEFAULT = 500 #500
OPTIMIZER_DEFAULT = 'ADAM'
SIZE_OUTPUT = 10

# Directory in which cifar data is saved
DATA_DIR_DEFAULT = './cifar10/cifar-10-batches-py'

FLAGS = None

OPT_PLOT = True

save_dir = './../../../saveData/'

def accuracy(predictions, targets):
  """
  Computes the prediction accuracy, i.e. the average of correct predictions
  of the network.
  
  Args:
    predictions: 2D float array of size [batch_size, n_classes]
    labels: 2D int array of size [batch_size, n_classes]
            with one-hot encoding. Ground truth labels for
            each sample in the batch
  Returns:
    accuracy: scalar float, the accuracy of predictions,
              i.e. the average correct predictions over the whole batch
  
  TODO:
  Implement accuracy computation.
  """

  ########################
  # PUT YOUR CODE HERE  #
  #######################
  utils_custom.accuracy(predictions, targets)
  ########################
  # END OF YOUR CODE    #
  #######################

  return accuracy

def train():
  """
  Performs training and evaluation of ConvNet model. 

  TODO:
  Implement training and evaluation of ConvNet model. Evaluate your model on the whole test set each eval_freq iterations.
  """

  ### DO NOT CHANGE SEEDS!
  # Set the random seeds for reproducibility
  np.random.seed(42)

  ########################
  # PUT YOUR CODE HERE  #
  #######################
  n_classes = SIZE_OUTPUT
  learning_rate = FLAGS.learning_rate
  max_steps = FLAGS.max_steps
  batch_size = FLAGS.batch_size
  eval_freq = FLAGS.eval_freq
  data_dir = FLAGS.data_dir


  # load cifar data
  cifar10 = cifar10_utils.get_cifar10(data_dir, one_hot=False)


  # initialize network
  net = ConvNet(3, 10)
  loss_module = nn.CrossEntropyLoss()
  optimizer = optim.Adam(net.parameters())

  # initialize lists
  train_losses = []
  test_losses = []
  train_accuracies = []
  test_accuracies = []

  epoch = 0
  eval_steps = []
  while epoch < max_steps:
    # reset gradients
    optimizer.zero_grad()

    # get next batch
    x, y = cifar10['train'].next_batch(batch_size)

    x = torch.tensor(x, requires_grad=True)
    y = torch.tensor(y, dtype=torch.long)

    # run forward
    out = net.forward(x)

    # compute loss
    loss = loss_module(out, y)

    # compute gradient and update weights
    loss.backward()
    optimizer.step()

    epoch += 1

    # check if accuracy needs to be evaluated:
    if (epoch % EVAL_FREQ_DEFAULT) == 0:

      eval_steps.append(epoch)
      print("Evaluating at epoch: " + str(epoch))

      temp_test_loss = []
      temp_test_accuracy = []


      # load test data
      x_test, y_test = cifar10['test'].next_batch(200)  # , cifar10['test'].labels
      x_test = torch.tensor(x_test)
      y_test = torch.tensor(y_test).type(torch.long)

      accuracy_train = utils_custom.accuracy(out, y)
      train_accuracies.append(accuracy_train)
      print("training accuracy: ")
      print(accuracy_train)

      train_losses.append(loss.data.numpy())

      out_test = net.forward(x_test)
      loss_test = loss_module(out_test, y_test)

      print("loss Test: ")
      print(loss_test)
      val_loss_test = loss_test.data.cpu().numpy()
      test_losses.append(val_loss_test)

      accuracy_test = utils_custom.accuracy(out_test, y_test)
      print("Testing accuracy: ")
      print(accuracy_test)
      test_accuracies.append(accuracy_test)

  #try:
  #torch.save(ConvNet.state_dict(), save_dir + 'model/' + 'LastConvNet.pt' )
  #except:
    #print("save Failed")

  if OPT_PLOT:
    str_save = 'convnet_pytorch_run1'

    utils_custom.plot_accuracies(train_losses, train_accuracies, test_losses, test_accuracies, eval_steps, str_save, save_dir, FLAGS)
    utils_custom.save_results(train_losses, train_accuracies, test_losses, test_accuracies, str_save, save_dir, FLAGS)
  ########################
  # END OF YOUR CODE    #
  #######################

def print_flags():
  """
  Prints all entries in FLAGS variable.
  """
  for key, value in vars(FLAGS).items():
    print(key + ' : ' + str(value))

def main():
  """
  Main function
  """
  # Print all Flags to confirm parameter settings
  print_flags()

  if not os.path.exists(FLAGS.data_dir):
    os.makedirs(FLAGS.data_dir)

  # Run the training operation
  train()

if __name__ == '__main__':
  # Command line arguments
  parser = argparse.ArgumentParser()
  parser.add_argument('--learning_rate', type = float, default = LEARNING_RATE_DEFAULT,
                      help='Learning rate')
  parser.add_argument('--max_steps', type = int, default = MAX_STEPS_DEFAULT,
                      help='Number of steps to run trainer.')
  parser.add_argument('--batch_size', type = int, default = BATCH_SIZE_DEFAULT,
                      help='Batch size to run trainer.')
  parser.add_argument('--eval_freq', type=int, default=EVAL_FREQ_DEFAULT,
                        help='Frequency of evaluation on the test set')
  parser.add_argument('--data_dir', type = str, default = DATA_DIR_DEFAULT,
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()

  main()