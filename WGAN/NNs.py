import torch
import torch.nn as nn
import numpy as np

class DiscriminatorMLP(nn.Module):

    def __init__(self, image_size, num_of_channels):
        super(DiscriminatorMLP, self).__init__()

        self.image_size = image_size
        self.num_of_channels = num_of_channels

        self.model = nn.Sequential(
            nn.Linear(self.num_of_channels * self.image_size ** 2, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        return self.model(x)

class DiscriminatorConvNet(nn.Module):

    def __init__(self, image_size, num_of_channels, ngpu=1):
        super(DiscriminatorConvNet, self).__init__()
        self.ngpu = ngpu
        self.image_size = image_size
        self.num_of_channels = num_of_channels
        self.main = self.layers()

    def layers(self):
        n_hidden_layers = np.int(np.log2(self.image_size) - 3)

        modules = []
        modules += [
              nn.Conv2d(self.num_of_channels, self.image_size, 4, 2, 1, bias=False),
              nn.LeakyReLU(0.2, inplace=True)
        ]

        for num in range(n_hidden_layers):
            n_input_channels = 2 ** num
            n_output_channels = 2 ** (num + 1)

            modules += [
                  nn.Conv2d(self.image_size * n_input_channels, self.image_size * n_output_channels, 4, 2, 1, bias=False),
                  nn.BatchNorm2d(self.image_size * n_output_channels),
                  nn.LeakyReLU(0.2, inplace=True)
            ]

        modules += [
              nn.Conv2d(self.image_size * n_output_channels, 1, 4, 1, 0, bias=False),
        ]

        return nn.Sequential(*modules)

    def forward(self, input):
        return self.main(input)