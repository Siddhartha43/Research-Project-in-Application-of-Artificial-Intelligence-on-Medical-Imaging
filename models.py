import torch.nn as nn
import torch

# TODO implement EEGNet model
class EEGNet(nn.Module):
    def __init__(self, activation_func = 'relu', dropout = 0.25):
        super(EEGNet, self).__init__()
        # Choose activation function
        if activation_func == 'relu':
            activation = nn.ReLU()
        elif activation_func == 'leaky_relu':
            activation = nn.LeakyReLU()
        elif activation_func == 'elu':
            activation = nn.ELU()
        else:
            raise ValueError(f"Invalid activation function: {activation_func}")
        
        self.firstconv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(1, 51), stride=(1, 1), padding=(0, 25), bias=False),
            nn.BatchNorm2d(16)
        )

        self.depthwiseConv = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=(2, 1), stride=(1, 1), groups=16, bias=False),
            nn.BatchNorm2d(32),
            activation,
            nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4), padding=0),
            nn.Dropout(p=dropout)
        )
        self.separableConv = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(1, 15), stride=(1, 1), padding=(0, 7), bias=False),
            nn.BatchNorm2d(32),
            activation,
            nn.AvgPool2d(kernel_size=(1, 8), stride=(1, 8), padding=0),
            nn.Dropout(p=dropout)
        )
        self.classify = nn.Sequential(
            nn.Linear(in_features=736, out_features=2, bias=True)
        )

    def forward(self, x):
        x = self.firstconv(x)
        x = self.depthwiseConv(x)
        x = self.separableConv(x)
        x = torch.flatten(x, 1)  # flatten all dimensions except the batch dimension
        x = self.classify(x)
        return x

# (Optional) implement DeepConvNet model
class DeepConvNet(nn.Module):
    def __init__(self, activation_func='relu', dropout=0.5):
        super(DeepConvNet, self).__init__()
        # Add a dictionary to map activation function names to actual functions
        self.activation_funcs = {
            'relu': nn.ReLU(),
            'leaky_relu': nn.LeakyReLU(),
            'elu': nn.ELU()
        }
        # Check if the specified activation function is available
        if activation_func not in self.activation_funcs:
            raise ValueError(f"Invalid activation function: {activation_func}")

        # Use the specified activation function
        self.activation_func = self.activation_funcs[activation_func]
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 25, (1, 5)),
            nn.Conv2d(25, 25, (2, 1)),
            nn.BatchNorm2d(25),
            self.activation_funcs[activation_func], # Use the activation function passed as argument
            nn.MaxPool2d((1, 2)),
            nn.Dropout(dropout),
            nn.Conv2d(25, 50, (1, 5)),
            nn.BatchNorm2d(50),
            self.activation_funcs[activation_func], # Use the activation function passed as argument
            nn.MaxPool2d((1, 2)),
            nn.Dropout(dropout),
            nn.Conv2d(50, 100, (1, 5)),
            nn.BatchNorm2d(100),
            self.activation_funcs[activation_func], # Use the activation function passed as argument
            nn.MaxPool2d((1, 2)),
            nn.Dropout(dropout),
            nn.Conv2d(100, 200, (1, 5)),
            nn.BatchNorm2d(200),
            self.activation_funcs[activation_func], # Use the activation function passed as argument
            nn.MaxPool2d((1, 2)),
            nn.Dropout(dropout)
        )
        self.fc = nn.Linear(8600, 2)

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(-1, 8600)
        output = self.fc(x)
        return output