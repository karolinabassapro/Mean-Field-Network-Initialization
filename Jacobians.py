import torch
from torch.autograd import grad

class Jacobian():
    """
    Class for Jacobian of a given model. Once the class is instantiated with a given model it can be called on input data to return the Jacobian wrt that input data. Everything is done in double precision.
    """
    def __init__(self, n_classes, is_square, model):
        """
        Inputs:
            n_classes: int, number of classes
            is_square: bool, will the input data be square or flat, rule of thumb: CNN is true and FF is false.
            model: class, the network itself
        """
        self.n_classes = n_classes
        self.model = model.double()
        self.is_square = is_square

    def __call__(self, data):
        """
        Compute the Jacobian of selfl.model with respect to the data passed in. This jacobian matrix is stored in self.Jacob_mat until this is called again.
        Inputs:
            data: torch.tensor
        """
        self.Jacob_mat = torch.zeros([self.n_classes, len(data)])
        # may need ot use data.shape[1] instead
        data = data[None, None, ...].double()
        # NNs expect a tensor of length 4 the first index is batch
        # size, second is num channels.
        # Should change the second None here to n_channels
        data.requires_grad_()
        if self.is_square:
            out = self.model(data)[0]
            self.Jacob_mat = torch.zeros([self.n_classes, len(data.flatten())])
        else:
            out = self.model(data)[0][0]
        
        for i in range(self.n_classes):
            single_grad = grad(out[i], data, retain_graph= True)[0]
            single_grad.squeeze_()
            # kill the first and second indices as above
            self.Jacob_mat[i,:] = single_grad.flatten()
        return self.Jacob_mat
    
    def SingVals(self):
        """
        Return the singular values of self.Jacob_mat
        """
        singular_values = torch.linalg.svdvals(self.Jacob_mat)
        return singular_values