import torch
import torch.nn as nn
class FFNet(nn.Module):
    def __init__(self, num_in, num_classes, hidden_width, depth):
        super().__init__()

        self.depth = depth

        self.inlayer = nn.Linear(num_in, hidden_width)

        self.layers = nn.ModuleList([nn.Linear(hidden_width, hidden_width) for i in range(self.depth - 1)])

        self.output = nn.Linear(hidden_width, num_classes)

    def forward(self, x):
        x = torch.tanh(self.inlayer(x))
        for  i, layer in enumerate(self.layers):
            x = torch.tanh(layer(x))

        x = self.output(x)
        return x