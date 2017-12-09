import torch
import sys
import torch.nn as nn
from torch.autograd import Variable

class RNNPytorch(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, learning_rate):
        super(RNNPytorch, self).__init__()

        self.learning_rate = learning_rate
        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, output_size)
        #self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax()

    def _combine_input_hidden(self, input, hidden):
        return torch.cat((input, hidden), 1)

    def forward(self, input, hidden):
        combined = self._combine_input_hidden(input, hidden)
        hidden = torch.nn.functional.relu(self.i2h(combined))
        return hidden

    def get_output(self, input, hidden):
        combined = self._combine_input_hidden(input, hidden)
        hidden = torch.nn.functional.relu(self.i2h(combined))
        output = self.h2o(hidden)
        output = self.softmax(output)
        return output


    def initHidden(self):
        return Variable(torch.zeros(1, self.hidden_size))

