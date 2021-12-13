from typing import Optional
import torch, math
from scipy.stats import ortho_group
import torch.nn as nn
from MeanField import *

class Initialization():
    def __init__(self, net_style, in_size_or_channels, q = 2, name = 'xavier', activation = 'tanh'):
        """
        This class allows for each type of initialization to be called for each type of activation function for both convolutional nets and feedforward nets. This was created in order to avoid errors when changing from one style of net to another, as was happening frequently.
        Inputs:
            net_style: str, "Conv" or "FF" to indicate whether to change the initialization of the convolutional or feedforward layers

            in_size_or_channels: int, the number of inputs in the FF case, or the number of channels times the width of the filter in the convolutional case (9 for this CNN)

            q: float, q parameterizes the critical line of weights and biases for both FF and Conv neural networks. See [https://arxiv.org/abs/1802.09979]

            name: str, either xavier, gaussian, or orthogonal for which type of initialization is desired.

            activation: str, either tanh, relu, or hard_tanh
        """
        if name not in ("xavier", "gaussian", "orthogonal"):
            raise ValueError("Only xavier, gaussian, and orthogonal inits are suppported")
        elif activation not in ("tanh", "relu", "hard_tanh"):
            raise ValueError("Only tanh, relu, and hard_tanh activations are supported")
        elif net_style not in ("Conv, FF"):
            raise ValueError("Only Conv and FF are supported")

        self.q = q
        self.activation = activation
        self.ins = in_size_or_channels
        self.bias = False

        if net_style == "Conv":
            self.layer_type = nn.Conv2d
        else:
            self.layer_type = nn.Linear
            if name == "Orthogonal":
                raise Exception("Orthogonal Init doesn't work with FF Net")

        if name == "Orthogonal":
            self.init_type = self.delta_orthogonal
        elif name == "gaussian":
            self.init_type = self.Gaussian
        else:
            self.init_type = torch.nn.init.xavier_normal_

    def __call__(self, layer):
        if isinstance(layer, self.layer_type):
            layer.weight.data = self.init_type(layer.weight.data)
            if self.bias == True:
                layer.bias.data = self.init_type(layer.bias.data)

    def delta_orthogonal(self, tensor, gain=1.):
        """
        Generate delta orthogonal kernel for cnn. Weight tensor should be
        out channel x in channel x 3 x 3. Following algorithm 2 in: https://arxiv.org/abs/1806.05393
        Inputs:
            tensor: torch.Tensor, channel_out x channel_in
            gain: float, multiplicative factor to apply default is 1.
        out:
            tensor: output random init
        """
        self.bias = False
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


    def Gaussian(self, tensor):
        sw, sb = self.Gauss_compute()
        tensor.zero_()
        if not self.bias:
            tensor = torch.normal(torch.zeros(tensor.shape), torch.sqrt(sw/self.ins * torch.ones(tensor.shape)))
        else:
            tensor = torch.normal(torch.zeros(tensor.shape), torch.sqrt(sb/self.ins * torch.ones(tensor.shape)))

        self.bias = True
        return tensor

    def Gauss_compute(self):
        if self.activation == "relu":
            phi = lambda x: np.maximum(x, 0.0)
            d_phi = lambda x: (x > 0).astype("int")
            mf = MeanField(phi, d_phi)
            sw, sb = mf.get_noise_and_var(self.q, 1)
        elif self.activation == "tanh":
            phi = math.tanh
            d_phi = d_tanh
            mf = MeanField(phi, d_phi)
            sw, sb = mf.get_noise_and_var(self.q, 1)
        else:
            phi = lambda x: np.maximum(-1.0, np.minimum(1.0, x))
            d_phi = lambda x: np.logical_and(x > -1.0, x < 1.0).astype("int")
            mf = MeanField(phi, d_phi)
            sw, sb = mf.get_noise_and_var(self.q, 1)
        return sw, sb

        

# def init_xavier(layer):
#     """
#     Helper function which can recursively apply the Xavier
#     initialization to every layer of a network
#     """
#     if isinstance(layer, nn.Conv2d):
#        layer.weight.data = torch.nn.init.xavier_normal_(layer.weight.data, gain=1.0)

# def init_Gaussian(layer):
#     """
#     name = "relu", "tanh", or "hard_tanh"
#     Helper function which can recursively apply the Gaussian
#      initialization to every layer of a network
#     """
#     if name not in ("relu", "tanh", "hard_tanh"):
#         raise ValueError("say 'relu', 'tanh', or 'hard_tanh' only")
#     if name == "relu":
#         phi = lambda x: np.maximum(x, 0.0)
#         d_phi = lambda x: (x > 0).astype("int")
#         mf = MeanField(phi, d_phi)
#         sw, sb = mf.get_noise_and_var(q, 1)
#     elif name == "tanh":
#         phi = math.tanh
#         d_phi = d_tanh
#         mf = MeanField(phi, d_phi)
#         sw, sb = mf.get_noise_and_var(q, 1)
#     else:
#         phi = lambda x: np.maximum(-1.0, np.minimum(1.0, x))
#         d_phi = lambda x: np.logical_and(x > -1.0, x < 1.0).astype("int")
#         mf = MeanField(phi, d_phi)
#         sw, sb = mf.get_noise_and_var(q, 1)

#     if isinstance(layer, nn.Conv2d):
#         layer.weight.data = torch.nn.init.normal_(layer.weight.data, mean=0, std = math.sqrt(sw/784))
#         layer.bias.data = torch.nn.init.normal_(layer.bias.data, mean = 0, std = math.sqrt(sb/784))