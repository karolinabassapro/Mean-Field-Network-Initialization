import torch
from torch.utils.data import DataLoader, random_split
import torchvision.datasets as datasets
import torchvision.transforms as transforms

batch_size = 500
num_workers = 8

def load_MNIST():
    """
    Load the MNIST dataset into dataloaders train_loader, val_loader, and test_loader with batch size 1 for SGD.
    
    Inputs:
        None
    
    Outputs:
        train/val/test_loader: pytorch dataloaders
        classes: tuple of classes
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307), (0.3081)),
        # this is approx the mean and std for mnist,
        # the explicit calculation has been deleted
        transforms.Pad(1)
        # this is needed to get proper dimensions in paper
    ])
    
    classes = ('0', '1', '2', '3','4', '5', '6', '7', '8','9')
    
    # load the training data and normalize and pad as above
    train_data = datasets.MNIST(root = './data', train = True, transform = transform, download = True)
    
    # split the train data into train-val
    train_data, val_data = random_split(train_data, [50_000, 10_000], generator=torch.Generator().manual_seed(42))

    test_data = datasets.MNIST(root = './data', train = False, transform = transform, download = True)
    
    # move the datasets into dataloaders
    train_loader = DataLoader(train_data, batch_size = batch_size, shuffle = True, num_workers = num_workers)
    val_loader = DataLoader(val_data, batch_size = batch_size, shuffle = False, num_workers = num_workers)
    test_loader = DataLoader(test_data, batch_size = batch_size, shuffle = False, num_workers = num_workers)
    
    return train_loader, val_loader, test_loader, classes
    
    
def load_CIFAR():
    """
    Load the CIFAR10 dataset into dataloaders train_loader, val_loader, and test_loader with batch size 1 for SGD.
    
    Inputs:
        None
    
    Outputs(in a tuple):
        train/val/test_loader: pytorch dataloaders
        classes: tuple of classes
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)), 
        # this is approx the mean and std for mnist,
        # the explicit calculation has been deleted
        transforms.Pad(1)
    ])
    
    classes = ('plane', 'car', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
    train_data = datasets.CIFAR10(root = './data', train = True, transform = transform, 
                            download = True)
    train_data, val_data = random_split(train_data, [40_000, 10_000], generator=torch.Generator().manual_seed(42))
    test_data = datasets.CIFAR10(root = './data', train = False, transform = transform, 
                            download = True)
    
    train_loader = DataLoader(train_data, batch_size = batch_size, shuffle = True, num_workers = num_workers)
    val_loader = DataLoader(val_data, batch_size = batch_size, shuffle = True, num_workers = num_workers)
    test_loader = DataLoader(test_data, batch_size = batch_size, shuffle = False, num_workers = num_workers)
    
    return train_loader, val_loader, test_loader, classes