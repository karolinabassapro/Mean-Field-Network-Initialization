import torch, math
from scipy.stats import ortho_group
import torch.nn as nn

def delta_orthogonal(tensor, gain=1.):
    """
    Generate delta orthogonal kernel for cnn. Weight tensor should be
    channel in x channel out. In channels must be < out channelsFollowing algorithm 2 in
    [Xiao et al., 2018]: https://arxiv.org/abs/1806.05393
    Inputs:
        tensor: torch.Tensor, 3x3xchannel_inxchannel_out
        gain: float, multiplicative factor to apply default is 1.
    out:
        tensor: output random init
    """
    if tensor.shape[3] != 3:
        raise ValueError("tensor should be 3 dim")
    
    if tensor.size(1) > tensor.size(0):
        raise ValueError("In_channels > Out_channels")
    
    # generate random ortho matrix
    H = torch.tensor(ortho_group.rvs(tensor.size(0)))
    
    H = H[:, :tensor.size(1)]
    with torch.no_grad():
        tensor.zero_()
        tensor[:,:,(tensor.size(2)-1)//2, (tensor.size(2)-1)//2] = H
        tensor.mul_(math.sqrt(gain))
    return tensor

def init_weights(layer):
    if isinstance(layer, nn.Conv2d):
        delta_orthogonal(layer.weight.data)