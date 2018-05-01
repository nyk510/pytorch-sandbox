# coding: utf-8
"""
モデルの訓練を実行するクラスの定義
"""
import torch
from torch.optim import SGD, Adam
import numpy as np
import torch.nn.functional as F

from tqdm import tqdm

__author__ = "nyk510"


class Trainer(object):
    """
    abstract class for trainer
    """

    def __init__(self, models, device=None):
        """
        :param tuple[torch.nn.Module] models: 学習中に重みが更新されるモデルの配列
        :param str device:
        """
        self.models = models

        if isinstance(device, str):
            if device == "cuda":
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = device

        self.callbacks = None

    def start_epoch(self, epoch):
        for model in self.models:
            model.to(self.device)
            model.train()

        for c in self.callbacks:
            c.initialize(self.models, self.device)
            c.start_epoch(epoch)

    def end_epoch(self, epoch, logs):
        for c in self.callbacks:
            c.end_epoch(epoch, logs)

    def fit(self, train_iterator, valid_iterator=None, epochs=10, callbacks=None, initial_lr=0.01,
            optimizer_params=None):
        """
        :param train_iterator:
        :param valid_iterator:
        :param int epochs:
        :param list[Callback] callbacks:
        :param float initial_lr:
        :param dict | None optimizer_params:
        :return:
        """
        if optimizer_params is None:
            optimizer_params = {}
        self.set_optimizers(lr=initial_lr, params=optimizer_params)
        self.callbacks = callbacks

        for epoch in range(1, epochs + 1):
            self.start_epoch(epoch)
            logs = self.train(epoch, train_iterator)
            self.end_epoch(epoch, logs)
        return self

    def set_optimizers(self, lr, params):
        raise NotImplementedError()

    def train(self, epoch, train_iterator):
        logs = {}
        count = 0
        total = len(train_iterator)
        for batch_idx, (x, _) in tqdm(enumerate(train_iterator), total=total):
            count += 1
            logs = self._forward_backward_core(x, _, logs)

        for k, v in logs.items():
            logs[k] /= count
        return logs

    def _forward_backward_core(self, x, t, logs):
        raise NotImplementedError()


class DCGANTrainer(Trainer):
    def __init__(self, generator, discriminator, device=None):
        """
        :param Generator generator:
        :param Discriminator discriminator:
        :param device:
        """
        super(DCGANTrainer, self).__init__(models=(generator, discriminator), device=device)
        self.generator = generator
        self.discriminator = discriminator
        self.opt_generator = None
        self.opt_discriminator = None

        self.callbacks = None

    def set_optimizers(self, lr, params):
        self.opt_generator = Adam(params=self.generator.parameters(),
                                 lr=lr, betas=(0.5, 0.999))
        self.opt_discriminator = Adam(params=self.discriminator.parameters(),
                                     lr=lr, betas=(0.5, 0.999))
        # self.opt_discriminator = SGD(self.discriminator.parameters(), lr=lr, **params)
        # self.opt_generator = SGD(self.generator.parameters(), lr=lr, **params)

    def _get_target(self, batch, ones=True):
        if ones:
            return torch.ones((batch, 1), device=self.device)
        else:
            return torch.zeros((batch, 1), device=self.device)

    def _get_random_z(self, batch):
        dim = self.generator.z_dim
        z = np.random.uniform(-1., 1., size=(batch, dim))
        z = torch.from_numpy(z).type(torch.FloatTensor).to(self.device)
        return z

    def reset_grad(self):
        for model in self.models:
            model.zero_grad()

    def _forward_backward_core(self, x, t, logs):
        """

        :param x:
        :param t:
        :param dict logs:
        :return:
        """
        batch_size_i = x.shape[0]
        ones = self._get_target(batch_size_i, ones=True)
        zeros = self._get_target(batch_size_i, ones=False)

        # train generator
        noize = self._get_random_z(batch_size_i)
        fakes = self.generator(noize)
        f_judges = self.discriminator(fakes)
        loss_generator = F.binary_cross_entropy(f_judges, target=ones)

        self.opt_generator.zero_grad()
        loss_generator.backward()
        self.opt_generator.step()

        # train discriminator
        noize = self._get_random_z(batch_size_i)
        fakes = self.generator(noize)
        f_judges = self.discriminator(fakes)

        loss_discriminator = F.binary_cross_entropy(f_judges,
                                                    target=zeros)
        x = x.to(self.device)
        t_judge = self.discriminator(x)
        loss_discriminator += F.binary_cross_entropy(t_judge,
                                                     target=ones)

        self.opt_discriminator.zero_grad()
        loss_discriminator.backward()
        self.opt_discriminator.step()

        for n, val in zip(("discriminator", "generator"), (loss_discriminator.item(), loss_generator.item())):
            if n in logs.keys():
                logs[n] += val
            else:
                logs[n] = val

        return logs
