import matplotlib.pyplot as plt
from MeanField import MeanField, d_tanh
from load_data import *
from Init_specifications import *
from Network import *
from MeanField import *
import torch.nn as nn
from numpy import tanh
from teams import FFNet
from Jacobians import Jacobian
from test_tools import *

def main():

    n = 100
    Q = 2
    
    mf = MeanField(math.tanh, d_tanh)
    h_0 = mf.get_h0(Q, 28).flatten()

    Jacobians = torch.zeros([n, 10, 784])
    max_sv = torch.zeros(n)

    init = Initialization("Conv",  784 ,name='gaussian', q = Q, activation = 'tanh')

    # for Depth in tqdm(range(n)):
    #     Net = FFNet(784, 10, 784, Depth)
    #     Net.apply(init)

    #     Net_Jacobian = Jacobian(10, False, Net)
    #     Jacobians[Depth, :, :] = Net_Jacobian(h_0)
    #     # print("Jacobian: ", Jacobians[Depth, :, :])
    #     max_sv[Depth] = torch.max(Net_Jacobian.SingVals())

    Net = test_net(784, 10)
    Net.apply(init)

    Net_Jacobian = Jacobian(10, False, Net)
    Net_Jacobian(h_0)
    singvals = Net_Jacobian.SingVals()

    print(torch.mean(singvals))

    x = np.arange(0, n)
    plt.hist(singvals,bins = 3)
    plt.show()


if __name__ == '__main__':
    main()