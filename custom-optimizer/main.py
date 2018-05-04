# coding: utf-8
"""
実行ファイル
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets import MNIST
from tqdm import tqdm

from optimizers import RDA, L1SGD

__author__ = "nyk510"


def load_binary_mnist(
        target_label_positive=6,
        target_label_negative=7):
    """
    MNIST データセットから
    特定のラベルを持つ subset のデータセットを作成する
    作成されたデータセットは二値分類問題のラベルが付いている
    すなわち positive に指定した画像に `+1`, negative に指定した画像に `0` のラベルがつく
    :param int target_label_positive: +1 のラベルをつける画像のクラス. [0, 9] の int
    :param int target_label_negative: 0 のラベルをつける画像のクラス [0, 9] の int
    :return: dataloader クラス.
    :rtype: DataLoader
    """
    class CustomDataset(TensorDataset):
        def __init__(self, x, t, transform=None):
            super().__init__(x, t)
            self.transform = transform

        def __getitem__(self, index):
            x, t = super().__getitem__(index)
            if self.transform is not None:
                x = self.transform(x)
            return x, t

    mnist_train = MNIST(root="../dataset/", train=True, download=True)

    train_data_positive = mnist_train.train_data[mnist_train.train_labels == target_label_positive]
    train_data_negative = mnist_train.train_data[mnist_train.train_labels == target_label_negative]
    train_x = torch.cat([train_data_positive, train_data_negative])
    train_labels = torch.cat([
        torch.ones(len(train_data_positive), 1),
        torch.zeros(len(train_data_negative), 1)
    ])

    # mnist の元が int (\in [0, 255]) なので float にしてないと model の行列計算でエラーになる
    # 元論文でスケーリングしていなかったので前処理はそれだけ
    train_dataset = CustomDataset(train_x, train_labels, transform=lambda x: x.type(dtype=torch.FloatTensor))

    # データはバッチ処理しないので batch-size は 1 にする
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    return train_loader


def train(model, optimizer, loader, device="cuda", epochs=2, zero_weight_threshold=1e-5):
    if device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    losses = []
    nonzero_nums = []
    model = model.to(device)
    objective = nn.BCELoss()
    for epoch in range(epochs):
        print("epoch: {epoch}".format(**locals()))
        model.train()
        loss_i = 0
        for batch_idx, (x, t) in tqdm(enumerate(loader), total=len(loader)):
            nonzero_nums.append(model.number_of_nonzero(zero_weight_threshold))
            x, t = x.to(device), t.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = objective(pred, t)
            loss.backward()
            optimizer.step()
            loss_i += loss.item()
        losses.append(loss_i / len(loader))
        print("loss:\t{}".format(losses[-1]))
    return losses, nonzero_nums


class LogisticRegression(nn.Module):
    """
    Logistic Regression Module
    """

    def __init__(self, input_dim=28, output_dim=1):
        super().__init__()
        self.flatten_dim = input_dim ** 2
        self.fc = nn.Linear(input_dim ** 2, output_dim)

    def forward(self, x):
        x = x.view(-1, self.flatten_dim)
        h = self.fc(x)
        h = F.sigmoid(h)
        return h

    def number_of_nonzero(self, threshold=1e-5):
        """
        重みの中で非ゼロのものを count する
        :param float threshold: しきい値. この値よりも絶対値の小さい重みをカウントする
        :return:
        """
        return (self.fc.weight.data.abs() > threshold).sum().item()


def main():
    train_loader = load_binary_mnist()
    df_loss = pd.DataFrame()
    df_nonzero = pd.DataFrame()

    lam = 10.
    gamma = 5e3
    # rda method
    model = LogisticRegression()
    optimizer = RDA(model.parameters(), lam=lam, gamma=gamma, rho=5e-3)
    losses, nonzeros = train(model, optimizer, train_loader)

    rda_weights = model.fc.weight.cpu().detach().numpy().reshape(28, 28)
    df_loss["rda"] = losses
    df_nonzero["rda"] = nonzeros

    # l1-sgd
    model = LogisticRegression()
    optimizer = L1SGD(model.parameters(), gamma=300., total=12000, lam=lam)
    losses, nonzeros = train(model, optimizer, train_loader)

    sgd_weight = model.fc.weight.cpu().detach().numpy().reshape(28, 28)
    df_loss["sgd"] = losses
    df_nonzero["sgd"] = nonzeros

    df_nonzero[:12000].plot()
    plt.title("nonzero values")
    plt.xlabel("iterations")
    plt.savefig("./nonzero_value_counts.png", dpi=150)

    fig = plt.figure(figsize=(10, 5))
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.imshow(rda_weights, interpolation=None, vmin=-1e-3, vmax=1e-3)
    ax1.set_title("RDA")
    ax1 = fig.add_subplot(1, 2, 2)
    ax1.imshow(sgd_weight, interpolation=None, vmin=-1e-2, vmax=1e-2)
    ax1.set_title("L1 SGD")
    fig.savefig("./weights_lambda={lam}.png".format(**locals()), dpi=150)


def weight_transition(lam_min=-2, lam_max=1, steps=5):
    lambdas = np.logspace(lam_min, lam_max, steps).tolist()[::-1]
    weights = []
    loader = load_binary_mnist()
    for lam in lambdas:
        print("start {lam:.2f}".format(**locals()))
        model = LogisticRegression()
        rda = RDA(model.parameters(), lam=lam)
        train(model, rda, loader)
        weights.append(model.fc.weight.cpu().detach().numpy().reshape(28, 28))

    fig = plt.figure(figsize=(len(lambdas) * 4, 4))
    for i, (w, lam) in enumerate(zip(weights, lambdas)):
        ax_i = fig.add_subplot(1, len(lambdas), i + 1)
        ax_i.imshow(w)
        ax_i.set_title("{lam:.2e}".format(**locals()))

    fig.savefig("./rda-weight-transition.png", dpi=150)


if __name__ == '__main__':
    # main()
    weight_transition()
