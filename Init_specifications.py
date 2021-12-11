import torch, math
from scipy.stats import ortho_group
import torch.nn as nn

def delta_orthogonal(tensor, gain=1.):
    """
    Generate delta orthogonal kernel for cnn. Weight tensor should be
    out channel x in channel x 3 x 3. Following algorithm 2 in: https://arxiv.org/abs/1806.05393
    Inputs:
        tensor: torch.Tensor, channel_out x channel_in
        gain: float, multiplicative factor to apply default is 1.
    out:
        tensor: output random init
    """
    if tensor.shape[3] != 3:
        print(tensor.shape)
        raise ValueError("tensor should be 3 dim")
    
    # generate random ortho matrix
    bigger = max(tensor.size(0), tensor.size(1))
    H = torch.tensor(ortho_group.rvs(bigger))
    
    # cut H to the appropriate size
    if tensor.size(0) > tensor.size(1):
        H = H[:, :tensor.size(1)]
    elif tensor.size(1) > tensor.size(0):
        H = H[:tensor.size(0), :]

    # set the initialization
    with torch.no_grad():
        tensor.zero_()
        tensor[:,:,(tensor.size(2)-1)//2, (tensor.size(2)-1)//2] = H
        tensor.mul_(math.sqrt(gain))
    return tensor

def init_ortho(layer):
    """
    This is a helper function which can recursively apply
    the delta orthogonalization to every layer of a network.
    """
    if isinstance(layer, nn.Conv2d):
        delta_orthogonal(layer.weight.data)

def init_xavier(layer):
    """
    Helper function which can recursively apply the Xavier
    initialization to every layer of a network
    """
    if isinstance(layer, nn.Conv2d):
        torch.nn.init.xavier_normal_(layer.weight.data, gain=1.0)