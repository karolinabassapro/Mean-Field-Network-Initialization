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
    # network = linear_128(in_width = 900, init_type = "gaussian")
    # network_jacob = Jacobian(network)
    # network_jacob(h_0)
    # singvals = network_jacob.SingVals().numpy()
    # print(singvals)
    # plt.hist(singvals)
    # plt.show()
    # print(np.mean(singvals))

    train_loader, val_loader, test_loader, classes = load_MNIST()
    print("1")
    num_runs = 10
    val_accs = pd.DataFrame(index=np.arange(num_runs), columns=["Gauss_val_accs","Xavier_val_accs","Ortho_val_accs"])
    print("2")

    for i in range(num_runs):
        print("3")
        gauss = linear_net(depth = 128, in_dim = 900, in_width = 900, init_type = "gaussian")
        print("4")
        xavier = linear_net(depth = 128, in_dim = 900, in_width = 900, init_type = "xavier")
        print("5")
        orthogonal = linear_net(depth = 128, in_dim = 900, in_width = 900, init_type = "orthogonal")
        print("6")
        df = pd.DataFrame(index=np.arange(40), columns=      ["Gauss_train_accs", "Xavier_train_accs",  "Ortho_train_accs"])

        _, accs_g = fit(gauss, 2, train_loader)
        print("7")
        _, accs_z = fit(xavier, 2, train_loader)
        print("8")
        _, accs_o = fit(orthogonal, 2, train_loader)
        print("9")

        for j in range(40):
            df.loc[j, "Gauss_train_accs"] = accs_g[j]
            df.loc[j, "Xavier_train_accs"] = accs_z[j]
            df.loc[j, "Ortho_train_accs"] = accs_o[j]
        print("grapes")

        val_accs.loc[i,"Gauss_val_accs"] = val_test(val_loader, gauss).item()
        val_accs.loc[i,"Xavier_val_accs"] = val_test(val_loader, xavier).item()
        val_accs.loc[i,"Ortho_val_accs"] = val_test(val_loader, orthogonal).item()

        df.to_csv(f'./results/data{i}.csv', index= False)

    val_accs.to_csv('./results/val_accs.csv', index=False)

    x = np.arange(len(accs_g))

    plt.plot(x, accs_g, label = "Gaussian")
    plt.plot(x, accs_z, label = "Xavier")
    plt.plot(x, accs_o, label = "Orthogonal")
    plt.title("Accuracies for Different Inits")
    plt.legend()
    plt.show()

    return

if __name__ == '__main__':
    main()
