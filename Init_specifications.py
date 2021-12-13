import torch, math
from scipy.stats import ortho_group
import torch.nn as nn
from MeanField import *

name = "hard_tanh"
q = 10

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

def init_Gaussian(layer):
    """
    name = "relu", "tanh", or "hard_tanh"
    Helper function which can recursively apply the Gaussian
     initialization to every layer of a network
    """
    if name not in ("relu", "tanh", "hard_tanh"):
        raise ValueError("say 'relu', 'tanh', or 'hard_tanh' only")
    if name == "relu":
        phi = lambda x: np.maximum(x, 0.0)
        d_phi = lambda x: (x > 0).astype("int")
        mf = MeanField(phi, d_phi)
        sw, sb = mf.get_noise_and_var(q, 1)
    elif name == "tanh":
        phi = math.tanh
        d_phi = d_tanh
        mf = MeanField(phi, d_phi)
        sw, sb = mf.get_noise_and_var(q, 1)
    else:
        phi = lambda x: np.maximum(-1.0, np.minimum(1.0, x))
        d_phi = lambda x: np.logical_and(x > -1.0, x < 1.0).astype("int")
        mf = MeanField(phi, d_phi)
        sw, sb = mf.get_noise_and_var(q, 1)

    if isinstance(layer, nn.Conv2d):
        torch.nn.init.normal_(layer.weight.data, mean=0, std = math.sqrt(sw/1024))
        torch.nn.init.normal_(layer.bias, mean = 0, std = math.sqrt(sb/1024))