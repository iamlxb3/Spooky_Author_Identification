import torch

class MLPPytorch(torch.nn.Module):
    def __init__(self):
        super(MLPPytorch, self).__init__()
        self.fc1 = torch.nn.Linear(784, 512)
        self.fc2 = torch.nn.Linear(512, 128)
        self.fc3 = torch.nn.Linear(128, 10)

    def forward(self, din):
        din = din.view(-1, 28 * 28)
        dout = torch.nn.functional.relu(self.fc1(din))
        dout = torch.nn.functional.relu(self.fc2(dout))
        return torch.nn.functional.softmax(self.fc3(dout))