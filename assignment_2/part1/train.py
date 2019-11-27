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
    np.random.seed(config.seed)
    assert config.model_type in ('RNN', 'LSTM')

    # Initialize the device which to run the model on
    device = torch.device(config.device)

    if config.model_type == 'RNN':
        # Initialize the model that we are going to use
        model = VanillaRNN(config.input_length, 
                           config.input_dim, 
                           config.num_hidden,
                           config.num_classes, 
                           device)  # fixme

    elif config.model_type == 'LSTM':
        model = LSTM(config.input_length, 
                           config.input_dim, 
                           config.num_hidden,
                           config.num_classes, 
                           device) 

    # Initialize the dataset and data loader (note the +1)
    dataset = PalindromeDataset(config.input_length+1)
    data_loader = DataLoader(dataset, config.batch_size, num_workers=1) # what is num_worker

    # Setup the loss and optimizer
    criterion = torch.nn.CrossEntropyLoss()  # (fix me) Dear TA's I can't fix you, you should talk to a therapist, please
    optimizer = torch.optim.RMSprop(model.parameters(), lr=config.learning_rate)  # (fixme), really please I am not qualified
    print(model.parameters())

    #init lists 
    accuracy_list = []
    accuracy_list_test = []
    loss_list = []
    loss_list_test = []
    epochs = []
    loss_prev = 0

    for step, (batch_inputs, batch_targets) in enumerate(data_loader):

        if debug: 
            print(('Input data 0: {0}; size: {1}').format(batch_inputs, batch_inputs.size()))
        # Only for time measurement of step through network
        t1 = time.time()

        # Add more code here ...
        
        #reset optimizer
        optimizer.zero_grad()

        
        # Add more code here ...
        #print(batch_inputs.size())
        #put input on target device 
        batch_inputs = batch_inputs.cuda(device)
        
        #transform to one-hot
        if config.input_dim == 10: 
            batch_inputs = torch.nn.functional.one_hot(batch_inputs.to(torch.int64), config.input_dim)
            batch_inputs = batch_inputs.to(torch.float)
            

        batch_targets = batch_targets.cuda(device)
        #get prediction, grads and update params
        out = model( batch_inputs )
        
        loss = criterion(out, batch_targets)   # (fixme) Your cries for help are getting distracting. I'm trying to finish this master succesfully
        #print(loss)
        loss.backward()

        ############################################################################
        # QUESTION: what happens here and why?
        ############################################################################
        torch.nn.utils.clip_grad_norm(model.parameters(), max_norm=config.max_norm)
        ############################################################################

        optimizer.step()

        accuracy = utils.calc_accuracy(out, batch_targets)  # (fixme) Okay listen: you just gotta be okay with who you are. focues on improving day by day. take baby steps and sometimes take a breath to appreciate how far you have come

        # Just for time measurement
        t2 = time.time()
        examples_per_second = config.batch_size/float(t2-t1)

        if step % 10 == 0:

            print("[{}] Train Step {:04d}/{:04f}, Batch Size = {}, Examples/Sec = {:.2f}, "
                  "Accuracy = {:.2f}, Loss = {:.3f}".format(
                    datetime.now().strftime("%Y-%m-%d %H:%M"), step,
                    config.train_steps, config.batch_size, examples_per_second,
                    accuracy, loss
            ))
            accuracy_list.append(accuracy)
            loss_list.append(loss)
            epochs.append(step)

        
        loss_change = loss - loss_prev
        
        if (step == config.train_steps) or ((loss_change >= -config.stop_criterium) and (loss_change <= 0)):
            # If you receive a PyTorch data-loader error, check this bug report:
            # https://github.com/pytorch/pytorch/pull/9655
            str_save = 'run'
            
            for key, value in vars(config).items(): 
                str_save += '__{0}_{1}'.format(key, value)
            print(str_save)

            #run a test set of 1000 samples
            test_data_loader = DataLoader(dataset, 1000, num_workers=1) # what is num_worker

            for step, (test_input, test_targets) in enumerate(test_data_loader):

                #move data to device
                test_input = test_input.cuda(device)
                
                #conver to one-hot
                if config.input_dim == 10: 
                    test_input = torch.nn.functional.one_hot(test_input.to(torch.int64), config.input_dim)
                    test_input = test_input.to(torch.float)

                test_targets = test_targets.cuda(device)

                #run test forward
                out = model(test_input)
                test_accuracy = utils.calc_accuracy(out, test_targets)
                print('Test accuracy: {}'.format(test_accuracy))
                test_accuracy = test_accuracy.cpu().numpy()
                break

            utils.plot_accuracies(loss_list, accuracy_list, test_accuracy, epochs, str_save=str_save, save_dir='../../../saveData/')
            
            break

        #save as previous loss 
        loss_prev = loss.clone().detach()
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
    parser.add_argument('--stop_criterium', type=float, default=0.0001, help="Stop criterum if loss change is below this value")
    parser.add_argument('--seed', type=float, default=42, help="Random seed for repeatability")


    config = parser.parse_args()

    # Train the model
    train(config)