from typing import Optional
from numpy.random import normal
import numpy as np
import torch, math
from scipy.stats import ortho_group
import torch.nn as nn
from MeanField import *

def gaussian(tensor, hidden_width):
    mf = MeanField(math.tanh,d_tanh)
    sw, sb = mf.get_noise_and_var(0, 1)
    std = math.sqrt(sw/hidden_width)
    mean = torch.zeros([tensor.shape[0], tensor.shape[1]])

    tensor = torch.normal(mean, torch.ones([tensor.shape[0], tensor.shape[1]]))
    tensor *= std

    return torch.nn.Parameter(tensor)

def gauss_bias(tensor):
    mf = MeanField(math.tanh,d_tanh)
    sw, sb = mf.get_noise_and_var(0, 1)
    std = torch.ones(tensor.shape[0])
    mean = torch.zeros(tensor.shape[0])
    std *= math.sqrt(sb)

    tensor = torch.nn.Parameter(torch.normal(mean, std))

    return tensor

def orthogonal(tensor):
    mf = MeanField(math.tanh,d_tanh)
    sw, sb = mf.get_noise_and_var(0, 1)
    bigger = max(tensor.shape[0], tensor.shape[1])
    a = tensor.new(bigger, bigger).normal_(0,1)
    q, r = torch.qr(a)
    d = torch.diag(r, 0)
    q *= d.sign()
    if tensor.size(0) > tensor.size(1):
        q = q[:, :tensor.size(1)]
    elif tensor.size(1) > tensor.size(0):
        q = q[:tensor.size(0), :]
    q *= math.sqrt(sw)
    tensor = torch.nn.Parameter(q.float())
    return tensor


def delta_orthogonal(tensor, gain = 1):
        """
        Generate delta orthogonal kernel for cnn. Weight tensor should be
        out channel x in channel x 3 x 3. Following algorithm 2 in: https://arxiv.org/abs/1806.05393
        Inputs:
            tensor: torch.Tensor, channel_out x channel_in
            gain: float, multiplicative factor to apply default is 1.
        out:
            tensor: output random init
        """

        if tensor.ndimension() < 3 or tensor.ndimension() > 5:
            raise ValueError("The tensor to initialize must be at least "
                       "three-dimensional and at most five-dimensional")
    
        if tensor.size(1) > tensor.size(0):
            raise ValueError("In_channels cannot be greater than out_channels.")
        
        # Generate a random matrix
        # a = tensor.new(tensor.size(0), tensor.size(0)).normal_(0, 1)
        # # Compute the qr factorization
        # q, r = torch.qr(a)
        # # Make Q uniform
        # d = torch.diag(r, 0)
        # q *= d.sign()
        # q = q[:, :tensor.size(1)]
        bigger = max(tensor.size(0), tensor.size(1))
        q = torch.tensor(ortho_group.rvs(bigger))
    
        # cut H to the appropriate size
        if tensor.size(0) > tensor.size(1):
            q = q[:, :tensor.size(1)]
        elif tensor.size(1) > tensor.size(0):
            q = q[:tensor.size(0), :]
        with torch.no_grad():
            tensor.zero_()
            if tensor.ndimension() == 3:
                tensor[:, :, (tensor.size(2)-1)//2] = q
            elif tensor.ndimension() == 4:
                tensor[:, :, (tensor.size(2)-1)//2, (tensor.size(3)-1)//2] = q
            else:
                tensor[:, :, (tensor.size(2)-1)//2, (tensor.size(3)-1)//2, (tensor.size(4)-1)//2] = q
            tensor.mul_(math.sqrt(gain))
            tensor = torch.nn.Parameter(tensor)
        return tensor
        
