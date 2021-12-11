from load_data import *
from Init_specifications import *
from Network import *

def main():
    cifar_train, cifar_val, cifar_test, cifar_classes = load_CIFAR()
    net = base_Conv(3, 10, 10, False)
    net.apply(init_weights) 
    fit(net, 1, cifar_train)

if __name__ == '__main__':
    main()