# coding: utf-8
"""
DCGAN で用いる Generator Discriminator の定義
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
    ランダムベクトルから shape = (1, 28, 28) の画像を生成するモジュール

    Note: はじめ Linear な部分には Batchnorm1d を入れていなかったが学習が Discriminator に全く追いつかず崩壊した。
    Generator のネットワーク構成はシビアっぽい
    """

    def __init__(self, z_dim=100):
        super().__init__()
        self.z_dim = z_dim

        self.flatten_dim = 5 ** 2 * 256
        self.fc = nn.Sequential(*[
            nn.Linear(z_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Linear(1024, self.flatten_dim),
            nn.BatchNorm1d(self.flatten_dim),
            nn.ReLU(True)
        ])

        self.upsample = nn.Sequential(
            make_layer(256, 128, kernel=3, stride=1, padding=0, deconv=True), # 5 -> 7
            make_layer(128, 64, kernel=4, stride=2, deconv=True), # 7 -> 14
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1) # 14 -> 28
        )

    def forward(self, input_tensor):
        h = self.fc(input_tensor)
        h = h.view(-1, 256, 5, 5)
        x = self.upsample(h)
        x = F.sigmoid(x)
        return x


class Discriminator(nn.Module):
    """
    shape = (1, 28, 28) の画像が入力された時に
    それが generator によって生成された画像かどうかを判別するモジュール

    web で実装を見ていると活性化関数に LeakyReLU を用いているものがありそちらのほうが学習が安定するらしい。
    今度活性化関数のみを変えて学習の安定度を見る等して差分を見てみたい。
    """

    def __init__(self):
        super().__init__()

        self.feature = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            make_layer(64, 128, kernel=4, stride=2),
            make_layer(128, 256, kernel=3, stride=1, padding=0)
        )

        # 入力画像は (1, 28, 28) で feature を forward してきた時には
        # 28 -> 14 -> 7 -> 5 になる
        self.flatten_dim = 5 ** 2 * 256
        self.fc = nn.Sequential(
            nn.Linear(self.flatten_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )

    def forward(self, input_tensor):
        h = self.feature(input_tensor)
        h = h.view(-1, self.flatten_dim)
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
    print(gen)

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
