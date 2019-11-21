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

        self.register_parameter('w_in', nn.Parameter(torch.randn(num_hidden, input_dim, device=device)))
        self.register_parameter('w_h',  nn.Parameter(torch.randn(num_hidden, num_hidden, device=device)))
        self.register_parameter('b_in', nn.Parameter(torch.randn(num_hidden, device=device)))

        #init weight matrix for output
        self.register_parameter('w_out', nn.Parameter(torch.randn(num_classes, num_hidden, device=device)))
        self.register_parameter('b_out', nn.Parameter(torch.randn(num_hidden, device=device)))

        #initialize a starting state
        self.hidden = torch.zeros(num_hidden, requires_grad=True, device=device)
        self.steps = seq_length
        print(self)

    def forward(self, x):
        # Implementation here ...
        debug = True
        #update hidden state
        input_hidden = torch.matmul(self.w_h, x)
        hidden_hidden = torch.matmul(self.w_h, self.hidden)
        if debug:
            print("Input to hidden proj: ")
            print(input_hidden.size())
            print("hidden (t-1) to hidden: ")
            print(hidden_hidden.size())
        self.hidden = nn.functional.tanh(input_hidden + hidden_hidden + self.b_in)
        
        #calculate output
        out = torch.matmul(self.hidden, self.w_out)  + self.b_out

        return out

