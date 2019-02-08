# coding: utf-8
"""
GPU vs CPU

GPU と CPU で実行時間がどのぐらい違うか見る

- on gpu
[time:] 123.194[s]

- on cpu
[time:] 381.905[s]

"""
from time import time

import torchvision
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

from nykertools.utils import stopwatch

transformer = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize([.5] * 3, [.5] * 3)]
)

train_set = torchvision.datasets.CIFAR10(root='/data/torchivision', train=True, download=True, transform=transformer)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

train_loader = DataLoader(train_set, batch_size=8, shuffle=True, num_workers=4)


class SimpleNetwork(nn.Module):
    def __init__(self):
        super(SimpleNetwork, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5, stride=1, padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 12, kernel_size=5, padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dense1 = nn.Linear(12 * 8 ** 2, 128)
        self.dense2 = nn.Linear(128, 10)

    def forward(self, input):
        x = self.pool1(F.relu(self.conv1(input)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(-1, 12 * 8 ** 2)
        x = F.relu(self.dense1(x))
        x = self.dense2(x)
        return x

@stopwatch
def train_network(device):
    net = SimpleNetwork().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=1e-2, momentum=.8, nesterov=True)

    n_epochs = 10

    for epoch in range(1, n_epochs + 1):
        run_loss = .0
        for i, (x_batch, y_label) in enumerate(train_loader, 0):
            optimizer.zero_grad()
            x_batch = x_batch.to(device)
            y_label = y_label.to(device)
            outputs = net(x_batch)
            loss = criterion(outputs, y_label)
            loss.backward()
            optimizer.step()

            run_loss += loss.item()

            if (i + 1) % 2000 == 0:
                run_loss /= 2000.
                print('[{epoch} {i}] loss: {run_loss:.3f}'.format(**locals()))
                run_loss = 0.


def main():
    for d in ['cuda:0', 'cpu']:
        train_network(device=d)


if __name__ == '__main__':
    main()
