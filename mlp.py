import torch.optim
import torch.nn as nn
import torch.utils.data
from torch.utils.data import TensorDataset, DataLoader
from typing import Sequence

torch.manual_seed(42)

class MLP(torch.nn.Module):
    NLS = {'relu': torch.nn.ReLU, 'tanh': nn.Tanh, 'sigmoid': nn.Sigmoid, 'softmax': nn.Softmax,
           'logsoftmax': nn.LogSoftmax, 'lrelu': nn.LeakyReLU}

    def __init__(self, D_in: int, hidden_dims: Sequence[int], D_out: int, nonlin='relu'):
        super().__init__()

        all_dims = [D_in, *hidden_dims, D_out]
        non_linearity = MLP.NLS[nonlin]
        layers = []

        for in_dim, out_dim in zip(all_dims[:-1], all_dims[1:]):
            layers += [
                nn.Linear(in_dim, out_dim, bias=True),
                non_linearity(),
                #nn.Dropout(0.25)

            ]

        # Sequential is a container for layers
        self.fc_layers = nn.Sequential(*layers[:-1])

        # Output non-linearity
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = torch.reshape(x, (x.shape[0], -1))
        z = self.fc_layers(x)
        y_pred = self.softmax(z)
        # Output is always probability
        return y_pred


def train_MLP(X,y):
    X_tensor = torch.tensor(X,dtype=torch.float)
    y_tensor = torch.tensor(y,dtype=torch.long)
    d = len(X_tensor[0])
    train_size = len(X_tensor)
    train_dataset = TensorDataset(X_tensor, y_tensor)  # create your datset
    dl_train = DataLoader(train_dataset, batch_size=4)
    print("dimension is ", d)
    model = MLP(D_in=d, hidden_dims=[781], D_out=2, nonlin='tanh')
    optimizer = torch.optim.SGD(params=model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()
    model.train()
    num_epochs = 3
    for_every = 1
    c = 0
    for epoch_idx in range(num_epochs):
        total_loss = 0
        num_correct = 0
        c += 1
        for batch_idx, (X, y) in enumerate(dl_train):
            # Forward pass
            y_prob = model(X)
            # Compute loss
            loss = loss_fn(y_prob, y)
            total_loss += loss.item()

            # Backward pass
            optimizer.zero_grad()  # Zero gradients of all parameters
            loss.backward()  # Run backprop algorithms to calculate gradients
            # Optimization step
            optimizer.step()  # Use gradients to update model parameters

            # Calc. number of correct char predictions
            y_pred = torch.argmax(y_prob, dim=1)
            num_correct += torch.sum(y_pred == y).float()

        if c % for_every == 0:
            print(
                f'Epoch #{epoch_idx + 1}: Avg. loss={total_loss / len(dl_train.dataset)}, Acc={num_correct / len(dl_train.dataset)}')
    return model