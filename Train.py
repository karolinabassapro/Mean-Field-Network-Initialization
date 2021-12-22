import torch, math
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from load_data import batch_size

num_between_report = 5

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
    num_right = 0
    
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay = decay)

    data = data.to(device)
    labels = label.to(device)
    
    # standard 5 step training
    optimizer.zero_grad()
    out = model(data)
    with torch.no_grad(): 
        # calculate the accuracy without touching gradient
        _, prediction = torch.max(out, 1)
        num_right += calculate_acc(prediction, label)
    loss = criterion(out, label)
    loss.backward()
    optimizer.step()
    
    return loss, num_right

def fit(model, epochs, train_loader, learning_rate = 0.01, decay = 0, criterion = nn.CrossEntropyLoss()):
    """
    Fit model and print out loss updates every 1000 training steps.


    """
    k = 0
    losses = np.zeros(epochs * math.ceil((len(train_loader)/(num_between_report))))
    accuracy = np.zeros(epochs * math.ceil((len(train_loader)/(num_between_report))))

    model = model.to(device)
    for epoch in tqdm(range(epochs)):
        loss_tracker = 0.0
        acc_tracker = 0

        model.train()

        for i, data in tqdm(enumerate(train_loader, 0)):
            train_in, label = data
            # flatten layers except batch
            train_in = torch.flatten(train_in, start_dim = 1)
            loss, acc = train_step(model, train_in, label, learning_rate, decay, criterion)
            loss_tracker += loss
            acc_tracker += acc

            # print out running accuracy and loss for every 1000 mini-batches
            if i % num_between_report == num_between_report - 1:
                #acc = val_test(val_loader, model)
                print(f"epoch: {epoch + 1}, image: {i + 1}, loss: {(loss_tracker / num_between_report)}, acc: {(acc_tracker/(num_between_report * batch_size))}")

                losses[k] = (loss_tracker / num_between_report)
                accuracy[k] = (acc_tracker  / (num_between_report * batch_size))
                k += 1
                loss_tracker = 0.0
                acc_tracker = 0

    print("Training Done")
    return losses, accuracy

def val_test(val_loader, model):
    """
    Test accuracy on the validation set
    Inputs:
        val_loader: dataloader of validation set
        model: neural network
    Outputs:
        acc: float, validation accuracy
    """
    num_correct = 0
    with torch.no_grad():
        for i, data in enumerate(val_loader, 0):
            train_in, label = data
            out = model(train_in)
            prediction = out.argmax(dim = 1)
            if prediction == label:
                num_correct += 1
    
    return num_correct/len(val_loader)

def calculate_acc(prediction, labels):
    """
    Helper function to calculate the number of correctly predicted labels.
    Inputs:
        prediction: torch.tensor, tensor of predictions from network
        labels: torch.tensor, true labels.
    """
    return len(prediction) - torch.count_nonzero(prediction - labels)