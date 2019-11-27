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

import torch
import torch.nn as nn


################################################################################

class VanillaRNN(nn.Module):

    def __init__(self, seq_length, input_dim, num_hidden, num_classes, device='cpu'):
        super(VanillaRNN, self).__init__()
        # Initialization here ...
        #init weight matrix for input

        self.register_parameter('w_in', nn.Parameter(torch.zeros(num_hidden, input_dim, device=device)))
        torch.nn.init.xavier_uniform_(self.w_in)
        #print('w_in weights with xavier init: {0}, with size: {1}'.format(self.w_in, self.w_in.size()))
        self.register_parameter('w_h',  nn.Parameter(torch.zeros(num_hidden, num_hidden, device=device)))
        torch.nn.init.xavier_uniform_(self.w_h)
        self.register_parameter('b_in', nn.Parameter(torch.zeros(num_hidden, 1, device=device)))

        #init weight matrix for output
        self.register_parameter('w_out', nn.Parameter(torch.zeros(num_hidden, num_classes, device=device)))
        torch.nn.init.xavier_uniform_(self.w_out)
        self.register_parameter('b_out', nn.Parameter(torch.zeros(num_classes, 1,  device=device)))

        #initialize a starting state
        self.hidden = torch.zeros(num_hidden, requires_grad=True, device=device)
        self.steps = seq_length
        

    def forward(self, x):
        
        #get batch size
        batch_size = x.size()[0]
        input_dim = x.dim()
        #expand hidden state to batch size
        hidden_cur = self.hidden.repeat(batch_size,1)

        for step in range(x.size()[1]):
            
            #print('Before transform: input size: {0}; input dimensions: {1}'.format(x.size(), x.dim()))
            if input_dim > 2: 
                cur_input = x[:, step].view(batch_size, 10).t()#x[:,step].t().view(-1, batch_size, x.size()[2])
            else: 
                cur_input = x[:, step].view(batch_size, 1).t()
            input_hidden = torch.matmul(self.w_in, cur_input)
            
            hidden_cur = nn.functional.tanh(input_hidden + hidden_hidden + self.b_in).t()
            
        #calculate output
        out = torch.matmul(hidden_cur, self.w_out) + self.b_out.view(1, -1)

        return out

