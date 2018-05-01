# coding: utf-8
"""
DCGAN の training を実行するメインスクリプト

TODO:コマンドライン引数での制御が全く実装できていないのであとでやること
"""

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import SGD
import torch.nn.functional as F
from torchvision import datasets, transforms
import numpy as np
from models import Generator, Discriminator
from callbacks import PrintCallback, PlotGenerator, DataframeLogger
from solver import DCGANSolver

__author__ = "nyk510"


def get_emnist_dataset():
    data_transformer = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize((.1,), (.3,)), # [-1, 1] の範囲にする
        transforms.Lambda(lambda x: np.transpose(x, (0, 2, 1)))  # x, y を反転させる
    ])
    emnist = datasets.EMNIST("../dataset/emnist", split="balanced", train=True, download=True,
                             transform=data_transformer)
    return emnist


def main():
    images = get_emnist_dataset()

    # このあたりはコマンドライン引数にしたい
    n_batch = 128
    epochs = 1000
    hidden_dim = 32
    device = "cuda"

    image_loader = DataLoader(images, batch_size=n_batch, shuffle=True, num_workers=8)

    generator = Generator(hidden_dim)
    discriminator = Discriminator()

    # SGD なら 1e-3, Adam なら 1e-4 オーダー程度が安定するようです
    initial_lr = .005

    trainer = DCGANSolver(generator, discriminator, device)

    # optimizer はなんとなく SGD with Nesterov
    optimizer_params = {
        "nesterov": True,
        "momentum": 0.8
    }

    # エポック前後に実行するコールバックを配列で定義。
    callbacks = [
        PrintCallback(),
        PlotGenerator(),
        DataframeLogger()
    ]
    trainer.fit(image_loader, epochs=epochs, initial_lr=initial_lr,
                callbacks=callbacks, optimizer_params=optimizer_params)


if __name__ == '__main__':
    main()
