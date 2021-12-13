import torch
import torch.nn as nn

def base_test(tensor, gain = 1):
    """
    This is just for testing
    """
    tensor = 17 * torch.eye(len(tensor))
    return tensor

def test_init(layer):
    """
    This is a helper function which can recursively apply
    the base test init to every layer of a network.
    """
    if isinstance(layer, nn.Linear):
        layer.weight.data = base_test(layer.weight.data)

class test_net(nn.Module):
    """
    An extremely simple 'network' for testing
    """
    def __init__(self, n_in, n_classes):
        """
        A testing class for the Jacobian
        """
        super().__init__()

        self.layer1 = nn.Linear(n_in, n_in)
        self.layer2 = nn.Linear(n_in, n_classes)

    def forward(self, x):
        x = torch.tanh(self.layer1(x))
        x = self.layer2(x)
        return x