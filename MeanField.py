import numpy as np
import torch
from scipy.stats import norm
from scipy.integrate import quad
import matplotlib.pyplot as plt
import warnings

def d_tanh(x):
    """Derivative of tanh."""
    return 1./ np.cosh(x)**2

class MeanField():
    """
    We closely follow the Xiao CNN paper and code here
    """
    def __init__(self, phi, dphi):
        """
        Inputs:
        phi: nonlinearity
        dphi: nonlinearity derivative
        """
        self.phi = phi
        self.dphi = dphi
    
    def q_pdf(self, x, q):
        """
        compute the q-map for q^l in terms of q^{l - 1} from Poole et al.
        """
        return (self.phi(np.sqrt(q)*x)**2) * norm.pdf(x)

    def chi_1_pdf(self, x, q):
        return (self.dphi(np.sqrt(q) * x)**2) * norm.pdf(x)

    def get_noise_and_var(self, q, chi):
        """
        get the variance (sigma^2) for the gaussian init of
        weight matrices (sw) and biases (sb)
        setting chi = 1 will give the critical line
        returns the critical line of sigmaw and sigmab
        """
        warnings.simplefilter("ignore")
        sw = chi/quad(self.chi_1_pdf, -np.inf, np.inf, args= (q))[0]
        sb = q - sw * quad(self.q_pdf, -np.inf, np.inf, args= (q))[0]
        return sw, sb

    def get_h0(self, q, n):
        if n <= 0:
            raise ValueError("Need dimension > 0")
        h0 = np.eye(n)
        return torch.tensor(np.sqrt(q)/n * h0)
        

    def plot(self):
        """
        Plot the phase transition diagram
        """
        background = [253/255.0, 236/255.0, 247/255.0]
        n = 50
        qrange = np.linspace(1e-5, 2.25, n)

        sw = [self.get_noise_and_var(q, 1)[0] for q in qrange]
        sb = [self.get_noise_and_var(q, 1)[1] for q in qrange]

        print(f"sw: {sw[0]}, sb : {sb[0]}")

        plt.figure(figsize=(5, 5))
        plt.plot(sw, sb)
        plt.xlim(0.5, 3)
        plt.ylim(0, 0.25)
        plt.title("Critical line")
        plt.xlabel("$\sigma_w^2$")
        plt.ylabel("$\sigma_b^2$")
        plt.gca().set_facecolor(background)
        plt.show()