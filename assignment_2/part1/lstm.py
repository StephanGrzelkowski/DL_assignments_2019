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
import torch.nn.functional as F

################################################################################

class LSTM(nn.Module):

    def __init__(self, seq_length, input_dim, num_hidden, num_classes, device='cpu'):
        super(LSTM, self).__init__()
        # Initialization here ...
        #init cell state 
        num_cell = num_hidden
        self.cell = {}
        self.cell.state = torch.zeros(num_hidden)

        #hidden layer
        self.hidden = {}
        self.hidden.state = torch.zeros(num_hidden)
        self.hidden.params = {}
        self.hidden.params['w'] = torch.randn(num_classes, num_hidden)
        self.hidden.params['b'] = torch.randn(num_classes) 
        #modulation gate
        self.modulation_gate = {}
        self.modulation_gate.params["w_x"] = torch.randn(num_cell, input_dim)
        self.modulation_gate.params["w_h"] = torch.randn(num_cell, num_hidden)
        self.modulation_gate.params['b'] = torch.randn(num_cell)

  		#input gate
        self.input_gate = {}
        self.input_gate.params["w_x"] = torch.randn(num_cell, input_dim)
        self.input_gate.params["w_h"] = torch.randn(num_cell, num_hidden)
        self.input_gate.params['b'] = torch.randn(num_cell)
  		
  		#forget gate
        self.forget_gate = {}
        self.forget_gate.params["w_x"] = torch.randn(num_cell, input_dim)
        self.forget_gate.params["w_h"] = torch.randn(num_cell, num_hidden)
        self.forget_gate.params['b'] = torch.randn(num_cell)
  		
  		#output gate 
        self.output_gate = {}
        self.output_gate.params["w_x"] = torch.randn(num_cell, input_dim)
        self.output_gate.params["w_h"] = torch.randn(num_cell, num_hidden)
        self.output_gate.params['b'] = torch.randn(num_cell)
  		
    def forward(self, x):

        # calculate all the gates
        g = F.tanh(self.modulation_gate.params['w_x'] * x + self.modulation_gate.params['w_h'] * self.hidden.state + self.modulation_gate.params['b']) 
        i = F.sigmoid(self.input_gate.params['w_x'] * x + self.input_gate.params['w_h'] * self.hidden.state, self.input_gate.params['b'])
        f = F.sigmoid(self.forget_gate.params['w_x'] * x + self.forget_gate.params['w_h'] * self.hidden.state, self.forget_gate.params['b'])
        o = F.sigmoid(self.output_gate.params['w_x'] * x + self.output_gate.params['w_h'] * self.hidden.state, self.output_gate.params['b'])
        
        #update cell state
        self.cell.state = torch.matmul(g, i) + torch.matmul(self.cell.state, f)
        
        #update hidden state
        self.hidden.state = torch.matmul(F.tanh(self.cell.state), o)

        #get output
        p = self.hidden.params['w'] * self.hidden.state + self.hidden.params['b']
        out = F.softmax(p)
