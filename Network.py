import torch
import torch.nn as nn
import Init_specifications as IS
class ff_net(nn.Module):
    def __init__(self, layer_gen, hidden_width, in_width, n_classes = 10, init_type = "orthogonal", bias = True):
        if init_type not in ["orthogonal", "gaussian", "xavier"]:
            print("Unsupported Init")
            init_type = "gaussian"
        super().__init__()
        self.layer_gen = layer_gen
        self.linear = nn.Linear(hidden_width, n_classes)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                if init_type == "orthogonal":
                    m.weight = IS.orthogonal(m.weight)
                    if bias:
                        m.bias = IS.gauss_bias(m.bias)
                elif init_type == "gaussian":
                    m.weight = IS.gaussian(m.weight, hidden_width)
                    if bias:
                        m.bias= IS.gauss_bias(m.bias)
                else:
                    nn.init.xavier_normal_(m.weight)


    def forward(self, x):
        x = self.layer_gen(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x

class conv_net(nn.Module):
    def __init__(self, layer_gen, hidden_channels, in_channels, n_classes = 10, init_type = "orthogonal"):
        if init_type not in ["orthogonal", "xavier"]:
            print("Unsupported init")
            init_type = "orthogonal"
        super(conv_net, self).__init__()
        self.layer_gen = layer_gen
        self.fc = nn.Linear(hidden_channels, n_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if init_type == "orthogonal":
                    m.weight = IS.delta_orthogonal(m.weight)
                    m.bias = IS.gauss_bias(m.bias)
                else:
                    nn.init.xavier_normal_(m.weight)
    
    def forward(self, x):
        x = self.layer_gen(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def make_conv(depth, in_channels):
    assert isinstance(depth, int)
    hidden_channels = 256 if depth <= 256 else 128
    layers = []
    for stride in [1,2,2]:
        conv2d = nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1, stride = stride)
        layers += [conv2d, nn.Tanh()]
        in_channels = hidden_channels
    for _ in range(depth):
        conv2d = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1)
        layers += [conv2d, nn.Tanh()]
    layers += [nn.AvgPool2d(8)] # 7 for mnist
    return nn.Sequential(*layers), hidden_channels

def make_linear(depth, in_width):
    hidden_width = 512 if depth < 256 else 256
    layers = []
    linear = nn.Linear(in_width, hidden_width, bias = True)
    layers += [linear, nn.Tanh()]
    for _ in range(depth):
        linear = nn.Linear(hidden_width, hidden_width, bias = True)
        layers += [linear, nn.Tanh()]
    return nn.Sequential(*layers), hidden_width

def conv_32(**kwargs):
    model = conv_net(*make_conv(32, 1), **kwargs)
    return model

def linear_net(depth, in_dim, **kwargs):
    model = ff_net(*make_linear(depth, in_dim), **kwargs)
    return model

def linear_128(**kwargs):
    model = ff_net(*make_linear(128, 3468), **kwargs)
    return model

def linear_256(**kwargs):
    model = ff_net(*make_linear(256, 3468), **kwargs)
    return model

def linear_512(**kwargs):
    model = ff_net(*make_linear(512, 3468), **kwargs)
    return model

def linear_1024(**kwargs):
    model = ff_net(*make_linear(1024, 3468), **kwargs)
    return model