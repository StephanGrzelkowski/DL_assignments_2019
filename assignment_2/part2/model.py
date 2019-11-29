# MIT License
#
# Copyright (c) 2019 Tom Runia
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

import torch.nn as nn
import torch

class TextGenerationModel(nn.Module):

    def __init__(self, batch_size, seq_length, vocabulary_size,
                 lstm_num_hidden=256, lstm_num_layers=2, device='cuda:0'):

        super(TextGenerationModel, self).__init__()
        # Initialization here...
        print('Init model hidden size: {}'.format(lstm_num_hidden))

        self.lstm = nn.LSTM( vocabulary_size, 
            lstm_num_hidden, 
            lstm_num_layers, 
            batch_first=True)

        self.fc = nn.Linear(lstm_num_hidden, vocabulary_size)
    
        self.to(device)

        self.batch_size = batch_size
        self.seq_length = seq_length
        self.vocabulary_size = vocabulary_size
        self.hidden_size = lstm_num_hidden
        self.device = device    

    def forward(self, x, temp=1):

        #let hidden and cell default to 0 (by leaving out)
        h = torch.zeros(2, self.batch_size, self.hidden_size, device=self.device)
        c = torch.zeros(2, self.batch_size, self.hidden_size, device=self.device)

        out = None 
        for step in range(self.seq_length):
            
            #get current time step 
            cur_input = x[:, step].view(self.batch_size, 1, self.vocabulary_size)

            
            #lstm forward 
            out_lstm, (h, c) = self.lstm(cur_input, (h, c))
            #linear layer
            out_linear = self.fc(out_lstm)
            #softmax
            out_softmax = torch.log_softmax(out_linear * temp, dim=2) #log cause cross entropy loss

            if out is None: 
                out = out_softmax
            else:
                out = torch.cat((out, out_softmax), 1)

            
        return out  