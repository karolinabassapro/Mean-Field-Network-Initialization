import matplotlib.pyplot as plt
import pandas as pd
from load_data import *
from Init_specifications import *
from Train import *
from Network import *
from MeanField import *
from Jacobians import Jacobian
from test_tools import *

def main():
    # mf = MeanField(math.tanh, d_tanh)
    # h_0 = mf.get_h0(0, 30, 512).flatten()
    # network = linear_32(in_width = 900, init_type = "gaussian")
    # network_jacob = Jacobian(network)
    # network_jacob(h_0)
    # singvals = network_jacob.SingVals().numpy()
    # print(singvals)
    # plt.hist(singvals)
    # plt.show()
    # print(np.mean(singvals))

    train_loader, val_loader, test_loader, classes = load_MNIST()

    gauss = linear_128(in_width = 900, init_type = "gaussian")
    xavier = linear_128(in_width = 900, init_type = "xavier")
    orthogonal = linear_128(in_width = 900, init_type = "orthogonal")

    gauss_losses, gauss_accuracies = fit(gauss, 2, train_loader)
    xavier_losses, xavier_accuracies = fit(xavier, 2, train_loader)
    ortho_losses, ortho_accuracies = fit(orthogonal, 2, train_loader)

    df = pd.DataFrame({ 
        "gauss_losses": gauss_losses, "gauss_acc": gauss_accuracies, "xavier_losses": xavier_losses, "xavier_accuracies": xavier_accuracies   
    })

    df.to_csv('./ data.csv', index= False)

    x = np.arange(len(gauss_accuracies))

    plt.plot(x, gauss_accuracies, label = "Gaussian")
    plt.plot(x, xavier_accuracies, label = "Xavier")
    plt.plot(x, ortho_accuracies, label = "Orthogonal")
    plt.title("Accuracies for Different Inits")
    plt.legend()
    plt.show()

    return

if __name__ == '__main__':
    main()