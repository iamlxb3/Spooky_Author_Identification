import torch as pt

class MLPPytorch(pt.nn.Module):
    def __init__(self):
        super(MLPPytorch, self).__init__()
        self.fc1 = pt.nn.Linear(784, 512)
        self.fc2 = pt.nn.Linear(512, 128)
        self.fc3 = pt.nn.Linear(128, 10)

    def forward(self, din):
        din = din.view(-1, 28 * 28)
        dout = pt.nn.functional.relu(self.fc1(din))
        dout = pt.nn.functional.relu(self.fc2(dout))
        return pt.nn.functional.softmax(self.fc3(dout))