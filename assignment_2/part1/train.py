################################################################################
# MIT License
# 
# Copyright (c) 2019
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course | Fall 2019
# Date Created: 2019-09-06
################################################################################

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import time
from datetime import datetime
import numpy as np

import torch
from torch.utils.data import DataLoader

"""To run from part1 folder""" 
import sys
sys.path.append("..")

from part1.dataset import PalindromeDataset
from part1.vanilla_rnn import VanillaRNN
from part1.lstm import LSTM
import utils

# You may want to look into tensorboard for logging
# from torch.utils.tensorboard import SummaryWriter

################################################################################
debug = False
def train(config):

    assert config.model_type in ('RNN', 'LSTM')

    # Initialize the device which to run the model on
    device = torch.device(config.device)

    # Initialize the model that we are going to use
    model = VanillaRNN(config.input_length, 
                       config.input_dim, 
                       config.num_hidden,
                       config.num_classes, 
                       device)  # fixme

    # Initialize the dataset and data loader (note the +1)
    dataset = PalindromeDataset(config.input_length+1)
    data_loader = DataLoader(dataset, config.batch_size, num_workers=1) # what is num_worker

    # Setup the loss and optimizer
    criterion = torch.nn.CrossEntropyLoss()  # (fix me) Dear TA's I can't fix you, you should talk to a therapist, please
    optimizer = torch.optim.RMSprop(model.parameters(), lr=config.learning_rate)  # (fixme), really please I am not qualified
    print(model.parameters())
    for step, (batch_inputs, batch_targets) in enumerate(data_loader):

        if debug: 
            print(('Input data 0: {0}; size: {1}').format(batch_inputs, batch_inputs.size()))
        # Only for time measurement of step through network
        t1 = time.time()

        # Add more code here ...
        
        #reset optimizer
        optimizer.zero_grad()

        ############################################################################
        # QUESTION: what happens here and why?
        ############################################################################
        torch.nn.utils.clip_grad_norm(model.parameters(), max_norm=config.max_norm)
        ############################################################################

        # Add more code here ...
        print(batch_inputs.size())
        #put input on target device 
        batch_inputs = batch_inputs.cuda(device)
        batch_targets = batch_targets.cuda(device)
        #get prediction, grads and update params
        out = model( batch_inputs )
        
        loss = criterion(out, batch_targets)   # (fixme) Your cries for help are getting distracting. I'm trying to finish this master succesfully
        print(loss)
        loss.backward()
        optimizer.step()

        accuracy = utils.calc_accuracy(out, batch_targets)  # (fixme) Okay listen: you just gotta be okay with who you are. focues on improving day by day. take baby steps and sometimes take a breath to appreciate how far you have come

        

        # Just for time measurement
        t2 = time.time()
        examples_per_second = config.batch_size/float(t2-t1)

        if step % 10 == 0:

            print("[{}] Train Step {:04d}/{:04d}, Batch Size = {}, Examples/Sec = {:.2f}, "
                  "Accuracy = {:.2f}, Loss = {:.3f}".format(
                    datetime.now().strftime("%Y-%m-%d %H:%M"), step,
                    config.train_steps, config.batch_size, examples_per_second,
                    accuracy, loss
            ))

        if step == config.train_steps:
            # If you receive a PyTorch data-loader error, check this bug report:
            # https://github.com/pytorch/pytorch/pull/9655
            break

    print('Done training.')


 ################################################################################
 ################################################################################

if __name__ == "__main__":

    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--model_type', type=str, default="RNN", help="Model type, should be 'RNN' or 'LSTM'")
    parser.add_argument('--input_length', type=int, default=10, help='Length of an input sequence')
    parser.add_argument('--input_dim', type=int, default=1, help='Dimensionality of input sequence')
    parser.add_argument('--num_classes', type=int, default=10, help='Dimensionality of output sequence')
    parser.add_argument('--num_hidden', type=int, default=128, help='Number of hidden units in the model')
    parser.add_argument('--batch_size', type=int, default=128, help='Number of examples to process in a batch')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--train_steps', type=int, default=10000, help='Number of training steps')
    parser.add_argument('--max_norm', type=float, default=10.0)
    parser.add_argument('--device', type=str, default="cuda:0", help="Training device 'cpu' or 'cuda:0'")

    config = parser.parse_args()

    # Train the model
    train(config)