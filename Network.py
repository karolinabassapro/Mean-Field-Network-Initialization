from re import L
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

class base_Conv(nn.Module):
    def __init__(self, n_channel, n_classes, depth, is_mnist):
        """
        Inputs:
            n_channel: int, number of input channels (1 for mnist, 3 for cifar)
            n_classes: int, number of input classes (10 for both mnist and cifar)
            depth: int, number of conv layers
            is_mnist: bool, indicate whether this is mnist or cifar
        """
        super().__init__()
        if is_mnist:
            self.dim_helper = 7 * 7
        else:
            self.dim_helper = 8 * 8
        self.depth = depth
        self.conv_stack = []
        self.conv1 = nn.Conv2d(n_channel, 64, 3, stride = 1, padding = "same")
        self.conv2 = nn.Conv2d(64, 128, 3, stride = 2)
        self.conv3 = nn.Conv2d(128, 256, 3, stride = 2, padding = 1)
        
        if self.depth < 0:
            raise ValueError("Need a positive depth")
        elif self.depth <= 256:
            self.many_convs1 = nn.ModuleList([nn.Conv2d(256, 256, 3, padding = "same")
                                            for i in range(self.depth - 3)])
            self.num_out_channels = 256
        else:
            self.many_convs1 = nn.ModuleList([nn.Conv2d(256, 256, 3, padding = "same")
                                            for i in range(253)])   
            self.many_convs2 = nn.Conv2d(256, 128, 3, padding = "same")
            self.many_convs3 = nn.ModuleList([nn.Conv2d(128, 128, 3, padding = "same")
                                            for i in range(254, self.depth - 3)])
            self.num_out_channels = 128
            
        self.fc = nn.Linear(self.num_out_channels * self.dim_helper, n_classes)

    def forward(self, x):
        x = torch.tanh(self.conv1(x))
        x = torch.tanh(self.conv2(x))
        x = torch.tanh(self.conv3(x))
        if self.depth <= 256:
            for i, layer in enumerate(self.many_convs1):
                x = torch.tanh(layer(x))
        else:
            for i, layer in enumerate(self.many_convs1):
                x = torch.tanh(layer(x))
            x = torch.tanh(self.many_convs2(x))
            for i, layer in enumerate(self.many_convs3):
                x = torch.tanh(layer(x))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = self.fc(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_step(model, data, label, learning_rate = 0.01, decay = 0, criterion = nn.CrossEntropyLoss()):
    '''
    Train one step for classification using SGD
    Inputs:
        model: NN
        data: torch.tensor, features of single training sample
        learning_rate: float,
        decay: float,
        criterion: torch.nn loss function

    Returns:
        loss: float
    '''
    
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay = decay)
    
    model = model.to(device)
    data = data.to(device)
    
    # standard 5 step training
    optimizer.zero_grad()
    out = model(data)
    loss = criterion(out, label)
    loss.backward()
    optimizer.step()
    
    return loss

def fit(model, epochs, train_loader, learning_rate = 0.01, decay = 0, criterion = nn.CrossEntropyLoss()):
    """
    Fit model and print out loss updates every 1000 training steps.


    """
    for epoch in tqdm(range(epochs)):
        loss_tracker = 0.0

        for i, data in enumerate(train_loader, 0):
            train_in, label = data
            loss_tracker += train_step(model, train_in, label, learning_rate, decay, criterion)

            if i % 1000 == 999:
                print(f"epoch: {epoch + 1}, image: {i + 1}, loss: {(loss_tracker / 1000)}")

                loss_tracker = 0.0

    print("Training Done")
