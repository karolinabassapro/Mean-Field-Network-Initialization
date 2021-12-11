from load_data import *
from Init_specifications import *
from Network import *
import numpy as np
import matplotlib.pyplot as plt

def main():
    # load cifar
    cifar_train, cifar_val, cifar_test, cifar_classes = load_CIFAR()

    # initialize the network with orthogonal weights
    orthogonal = base_Conv(n_channel = 3, n_classes = 10, depth = 10, is_mnist = False, size_1 = 32, size_2 = 16)
    orthogonal.apply(init_ortho) 

    losses, accuracies = fit(orthogonal, 1, cifar_train, learning_rate= 0.05)

    # initialize the network with Xavier initialization
    xavier = base_Conv(n_channel = 3, n_classes = 10, depth = 10, is_mnist = False, size_1 = 32, size_2 = 12)
    xavier.apply(init_xavier)

    losses2, accuracies2 = fit(xavier, 1, cifar_train, learning_rate= 0.05)

    x = np.arange(len(losses))
    plt.plot(x, losses, label="Ortho losses")
    plt.plot(x, accuracies, label = "Ortho accuracies")
    plt.plot(x, losses2, label="Xavier losses")
    plt.plot(x, accuracies2, label = "Xavier accuracies")
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()