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
        self.num_cell = num_hidden
        self.num_hidden = num_hidden
        self.device = device

        #REGISTER ALL PARAMETERS AND INITIALIZE XAVIER DISTRIBUTION
        #gate modulation 
        self.register_parameter('w_gx', nn.Parameter(torch.zeros(self.num_cell, input_dim, device=device)))
        torch.nn.init.xavier_uniform_(self.w_gx)
        self.register_parameter('w_gh', nn.Parameter(torch.zeros(self.num_cell, num_hidden, device=device)))
        torch.nn.init.xavier_uniform_(self.w_gh)
        self.register_parameter('b_g', nn.Parameter(torch.ones(self.num_cell, 1, device=device)))
        torch.nn.init.xavier_uniform_(self.b_g)
        
        #input gate 
        self.register_parameter('w_ix', nn.Parameter(torch.zeros(self.num_cell, input_dim, device=device)))
        torch.nn.init.xavier_uniform_(self.w_ix)
        self.register_parameter('w_ih', nn.Parameter(torch.zeros(self.num_cell, num_hidden, device=device)))
        torch.nn.init.xavier_uniform_(self.w_ih)
        self.register_parameter('b_i', nn.Parameter(torch.ones(self.num_cell, 1, device=device)))
        torch.nn.init.xavier_uniform_(self.b_i)
        
  		
  		#forget gate
        self.register_parameter('w_fx', nn.Parameter(torch.zeros(self.num_cell, input_dim, device=device)))
        torch.nn.init.xavier_uniform_(self.w_fx)
        self.register_parameter('w_fh', nn.Parameter(torch.zeros(self.num_cell, num_hidden, device=device)))
        torch.nn.init.xavier_uniform_(self.w_fh)
        self.register_parameter('b_f', nn.Parameter(torch.ones(self.num_cell, 1, device=device)))
        torch.nn.init.xavier_uniform_(self.b_f)
  		
  		#output gate 
        self.register_parameter('w_ox', nn.Parameter(torch.zeros(self.num_cell, input_dim, device=device)))
        torch.nn.init.xavier_uniform_(self.w_ox)
        self.register_parameter('w_oh', nn.Parameter(torch.zeros(self.num_cell, num_hidden, device=device)))
        torch.nn.init.xavier_uniform_(self.w_oh)
        self.register_parameter('b_o', nn.Parameter(torch.ones(self.num_cell, 1, device=device)))
        torch.nn.init.xavier_uniform_(self.b_o)
  		
        #out weights and bias 
        self.register_parameter('w_out', nn.Parameter(torch.zeros(num_classes, num_hidden, device=device)))
        torch.nn.init.xavier_uniform_(self.w_out)
        self.register_parameter('b_out', nn.Parameter(torch.ones(num_classes, 1, device=device)))
        
    def forward(self, x):
        
        #get relevant input dimensions
        batch_size = x.size()[0]
        seq_length = x.size()[1]
        input_dim = x.dim()
        
        #init hidden and cell state 
        hidden_state = torch.zeros(self.num_hidden, batch_size, device=self.device)
        cell_state = torch.zeros(self.num_cell, batch_size, device=self.device)

        for step in range(seq_length):
            if input_dim > 2:

                cur_input = x[:,step].view(batch_size, 10).t()
            else: 
                cur_input = x[:,step].view(batch_size, 1).t()

            # calculate all the gates
            g = F.tanh(torch.matmul(self.w_gx, cur_input) + torch.matmul(self.w_gh, hidden_state) + self.b_g) 
            i = F.sigmoid(torch.matmul(self.w_ix, cur_input) + torch.matmul(self.w_ih, hidden_state) +  self.b_i)
            f = F.sigmoid(torch.matmul(self.w_fx, cur_input) + torch.matmul(self.w_fh, hidden_state) +  self.b_f)            
            o = F.sigmoid(torch.matmul(self.w_ox, cur_input) + torch.matmul(self.w_oh, hidden_state) + self.b_o)
            
            #update cell state
            cell_state = g * i + cell_state * f
        
            #update hidden state
            hidden_state = F.tanh(cell_state) * o 
                
        #get output
        p = torch.matmul(self.w_out, hidden_state) + self.b_out
        
        out = F.softmax(p, dim=0).t() 

        return out
