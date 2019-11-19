"""
This module implements training and evaluation of a multi-layer perceptron in NumPy.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from mlp_numpy import MLP
from modules import CrossEntropyModule
import cifar10_utils
import utils_custom

# Default constants
DNN_HIDDEN_UNITS_DEFAULT = '100'
LEARNING_RATE_DEFAULT = 2e-3
MAX_STEPS_DEFAULT = 1 #1500
MAX_STEPS_DEFAULT = 1 #1500
BATCH_SIZE_DEFAULT = 200
EVAL_FREQ_DEFAULT = 100
NEG_SLOPE_DEFAULT = 0.02

# Directory in which cifar data is saved
DATA_DIR_DEFAULT = './cifar10/cifar-10-batches-py'

FLAGS = None


#do we wanna make new plots?
OPT_PLOT = True
SIZE_INPUT = 3 * 32 * 32
SIZE_OUTPUT = 10
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
  Performs training and evaluation of MLP model. 

  TODO:
  Implement training and evaluation of MLP model. Evaluate your model on the whole test set each eval_freq iterations.
  """

  ### DO NOT CHANGE SEEDS!
  # Set the random seeds for reproducibility
  np.random.seed(42)

  ## Prepare all functions
  # Get number of units in each hidden layer specified in the string such as 100,100
  if FLAGS.dnn_hidden_units:
    dnn_hidden_units = FLAGS.dnn_hidden_units.split(",")
    dnn_hidden_units = [int(dnn_hidden_unit_) for dnn_hidden_unit_ in dnn_hidden_units]
  else:
    dnn_hidden_units = []

  # Get negative slope parameter for LeakyReLU
  neg_slope = FLAGS.neg_slope

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
  cifar10 = cifar10_utils.get_cifar10(data_dir)

  # load test data
  x_test, y_test = cifar10['test'].images, cifar10['test'].labels
  x_test = x_test.reshape((-1, SIZE_INPUT))


  # initialize network
  net = MLP(SIZE_INPUT, dnn_hidden_units, n_classes, neg_slope)

  # initialize lists
  train_losses = []
  test_losses = []
  train_accuracies = []
  test_accuracies = []

  epoch = 0
  eval_steps = []
  while epoch < max_steps:
    # reset gradients


    # get next batch
    x, y = cifar10['train'].next_batch(batch_size)

    x = x.reshape((batch_size, SIZE_INPUT))  # THIS MIGHT BE WRONG

    # run forward
    out = net.forward(x)

    # compute loss
    loss = loss_module(out, y)

    """ COMPUTE GRADIENT AND UPDATE"""

    epoch += 1

    # check if accuracy needs to be evaluated:
    if (epoch % EVAL_FREQ_DEFAULT) == 0:
      eval_steps.append(epoch)
      print("Evaluating at epoch: " + str(epoch))

      accuracy_train = accuracy(out, y)
      train_accuracies.append(accuracy_train)
      print("training accuracy: ")
      print(accuracy_train)

      train_losses.append(loss.data.numpy())

      out_test = net.forward(x_test)
      loss_test = loss_module(out_test, y_test)

      val_loss_test = loss_test.data.cpu().numpy()
      test_losses.append(val_loss_test)

      accuracy_test = accuracy(out_test, y_test)
      print("Testing accuracy: ")
      print(accuracy_test)
      test_accuracies.append(accuracy_test)

  if OPT_PLOT:
    str_save = 'mlp_pytorch_run1'

    utils_custom.plot_accuracies(train_losses, train_accuracies, test_losses, test_accuracies, eval_steps, str_save,
                                 save_dir, FLAGS)
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
  parser.add_argument('--dnn_hidden_units', type = str, default = DNN_HIDDEN_UNITS_DEFAULT,
                      help='Comma separated list of number of units in each hidden layer')
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
  parser.add_argument('--neg_slope', type=float, default=NEG_SLOPE_DEFAULT,
                      help='Negative slope parameter for LeakyReLU')
  FLAGS, unparsed = parser.parse_known_args()

  main()