
import torch
import torch.nn as nn


class Net(nn.Module):
    def __init__(self, obs_size, hidden_size, n_actions):
        super(Net, self).__init__()
        # self.net = nn.Sequential(
        #     nn.Linear(obs_size, hidden_size),
        #     nn.ReLU(),
        #     nn.Linear(hidden_size, n_actions),
        # )
        self.conv1 = nn.Conv2d(7, 16, kernel_size=2, stride=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(16, 32, kernel_size=2, stride=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=1)
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(32*4*4, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions),
        )

    def forward(self, x):
        # flatten x
        # x = x.view(x.size(0), -1)
        # x = x.unsqueeze(1)
        # print("x shape", x.shape)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.maxpool2(x)
        x = self.flatten(x)
        # print("x shape", x.shape)
        x = self.fc(x)

        return x