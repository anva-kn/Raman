import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleDNN(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(SimpleDNN, self).__init__()
        self.conv1 = nn.Conv1D(n_input, n_hidden)
        self.activ1 = nn.ReLU()
        self.conv2 = nn.Conv1D()
        self.activ2 = nn.ReLU()
        self.mxpool1 = nn.MaxPool1d()
        self.drop1 = nn.Dropout()
        self.conv3 = nn.Conv1D()
        self.conv4 = nn.Conv1D()
        self.mxpool2 = nn.MaxPool1d()
        self.drop2 = nn.Dropout()
        self.conv5 = nn.Conv1D()
        self.conv6 = nn.Conv1D()
        self.mxpool3 = nn.MaxPool1d()
        self.flat1 = nn.Flatten()
        self.lin1 = nn.Linear()
        self.lin2 = nn.Linear(n_hidden, n_out)
        self.activ3 = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.activ2(x)
        x = self.conv2(x)
        x = self.activ2(x)
        x = self.mxpool1(x)
        x = self.drop1(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.mxpool2(x)
        x = self.drop2(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.mxpool3(x)
        x = self.flat1(x)
        x = self.lin1(x)
        x = self.activ3(self.lin2(x))
        return x