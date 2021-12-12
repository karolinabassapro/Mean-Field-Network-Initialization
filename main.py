from MeanField import MeanField, d_tanh
from load_data import *
from Init_specifications import *
from Network import *
from MeanField import *
import torch.nn as nn

def main():
    #load cifar
    cifar_train, cifar_val, cifar_test, cifar_classes = load_CIFAR()

    # initialize the network with orthogonal weights
    orthogonal = base_Conv(n_channel = 3, n_classes = 10, depth = 10, is_mnist = False, size_1 = 16, size_2 = 16)
    orthogonal.apply(init_ortho) 

    losses, accuracies = fit(orthogonal, 1, cifar_train, learning_rate= 0.05)

    # initialize the network with Xavier initialization
    xavier = base_Conv(n_channel = 3, n_classes = 10, depth = 10, is_mnist = False, size_1 = 16, size_2 = 16)
    xavier.apply(init_xavier)

    losses2, accuracies2 = fit(xavier, 1, cifar_train, learning_rate= 0.05)

    Gaussian = base_Conv(n_channel = 3, n_classes = 10, depth = 10, is_mnist = False, size_1 = 16, size_2 = 16)
    Gaussian.apply(init_Gaussian)

    losses3, accuracies3 = fit(Gaussian, 1, cifar_train, learning_rate= 0.05)

    x = np.arange(len(accuracies))
    plt.plot(x, accuracies, label = "Orthogonal accuracies")
    plt.plot(x, accuracies2, label = "Xavier accuracies")
    plt.plot(x, accuracies3, label = "Gaussian accuracies")
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()