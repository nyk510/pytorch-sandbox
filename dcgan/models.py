# coding: utf-8
"""
define dcgan generator and discriminator
"""

import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import ResNet

__author__ = "nyk510"


def make_layer(in_channels, out_channels, kernel=2, stride=2, padding=1, use_batchnorm=True, deconv=False):
    layers = []

    if deconv:
        conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel, stride=stride, padding=padding)
    else:
        conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel, stride=stride, padding=padding)

    layers.append(conv)
    if use_batchnorm:
        bn = nn.BatchNorm2d(out_channels)
        layers.append(bn)
    activate = nn.ReLU(True)
    layers.append(activate)

    return nn.Sequential(*layers)


class Generator(nn.Module):
    """
    ランダムベクトルから画像を生成する genearator
    """

    def __init__(self, z_dim=100):
        super().__init__()
        self.z_dim = z_dim

        self.fc = nn.Sequential(*[
            nn.Linear(z_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Linear(1024, 7 * 7 * 128),
            nn.BatchNorm1d(7 * 7 * 128),
            nn.ReLU(True)
        ])

        self.upsamples = nn.Sequential(
            make_layer(128, 64, kernel=3, stride=2, deconv=True),
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2)
        )

    def forward(self, input_tensor):
        h = self.fc(input_tensor)
        h = h.view(-1, 128, 7, 7)
        x = self.upsamples(h)
        x = F.sigmoid(x)
        return x


class Discriminator(nn.Module):
    """
    (1, 28, 28) の画像が入力された時にそれが genenator によって生成された画像かどうかを判別するモデル
    """

    def __init__(self):
        super().__init__()
        layers = []

        self.upsamles = nn.Sequential(
            nn.Conv2d(1, 128, kernel_size=4, stride=2),
            nn.LeakyReLU(0.2, inplace=True),
            make_layer(128, 256, kernel=3, stride=2)
        )
        self.fc = nn.Sequential(
            nn.Linear(7 * 7 * 256, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )

    def forward(self, input_tensor):
        h = self.upsamles(input_tensor)
        h = h.view(-1, 7 * 7 * 256)
        pred = self.fc(h)
        return pred


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import numpy as np
    nn.ConvTranspose2d(10, 20, 3)
    gen = Generator()
    dis = Discriminator()
    z = torch.randn(16, 100, dtype=torch.float)
    x = gen(z)
    print(x.shape)

    fig = plt.figure()
    for i in range(16):
        ax = fig.add_subplot(4, 4, i + 1)
        _x = x.detach().numpy().copy()[i][0]
        print(_x.shape)
        ax.imshow(_x)
    fig.tight_layout()
    plt.show()

    pred = dis(x)
    print(pred)
