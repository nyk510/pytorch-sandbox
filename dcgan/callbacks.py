# coding: utf-8
"""
学習中に呼び出されるコールバック関数の定義
keras の callback てきな奴
"""

import torch
from torchvision.utils import make_grid
import numpy as np
import matplotlib.pyplot as plt
import os
from models import Generator

__author__ = "nyk510"


class Callback(object):
    """
    abstract class for callback object
    this is called from trainer class when start epoch, end epoch,...
    """

    def __init__(self):
        self.models = None
        self.device = None
        pass

    def initialize(self, models, device):
        self.models = models
        self.device = device

    def start_train(self):
        pass

    def start_epoch(self, epoch, **params):
        pass

    def end_epoch(self, epoch, logs):
        pass

    def start_batch(self, batch_num, total, **params):
        pass

    def end_batch(self, batch_num, total, **params):
        pass


class PrintCallback(Callback):
    """
    標準出力に経過をプリントする.
    Todo: print ではなく logger.info などにする
    """
    def start_epoch(self, epoch, **params):
        print("start epoch\t{epoch}".format(**locals()))

    def end_epoch(self, epoch, logs):
        print("end epoch\t{epoch}".format(**locals()))
        print(logs)


class DataframeLogger(Callback):
    def __init__(self, out_file="./logs.csv"):
        import pandas as pd
        super(DataframeLogger, self).__init__()
        self.out_file = out_file
        self.out_dir = os.path.dirname(out_file)
        if os.path.exists(self.out_dir) is False:
            os.makedirs(self.out_dir)
        self.df = pd.DataFrame()

    def end_epoch(self, epoch, logs):
        logs["epoch"] = epoch
        self.df = self.df.append(logs, ignore_index=True)

        self.df.to_csv(self.out_file)



def convert_image_np(inp):
    """Convert a Tensor to numpy image."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    return inp

class PlotGenerator(Callback):
    def __init__(self, num_samples=64, output_dir="./visualize"):
        super(PlotGenerator, self).__init__()
        self.num_samples = num_samples
        self.output_dir = output_dir
        if os.path.exists(output_dir) is False:
            os.makedirs(output_dir)

    def end_epoch(self, epoch, logs):
        gen = None
        for model in self.models:
            if isinstance(model, Generator):
                gen = model

        if gen is None:
            return

        z_random = self.get_random_z(gen.z_dim, self.num_samples)
        z_gradual = self.get_gradual_z(gen.z_dim, self.num_samples)

        for z, name in zip((z_gradual, z_random), ("gradual", "random")):
            x = self.generate(gen, z)

            # 正規化
            x = x.sub(.1).div(.3)

            grids = make_grid(x)
            grids = convert_image_np(grids)

            fig = plt.figure(figsize=(8, 8))
            axis = fig.subplots()
            axis.imshow(grids)

            fname = "{name}_{epoch:04d}.png".format(**locals())
            fpath = os.path.join(self.output_dir, fname)
            fig.savefig(fpath, dpi=150)
            plt.close("all")

    def get_random_z(self, z_dim, length=32):
        """
        return: torch.Tensor: shape = (length, z_dim)
        """
        z = np.random.uniform(-1, 1, size=(length, z_dim))
        z = torch.from_numpy(z).type(torch.FloatTensor).to(self.device)
        return z

    def get_gradual_z(self, z_dim, length):
        """
        z_dim のベクトル `a` から別の z_dim のベクトル `b` との `length` の内分点を配列で返す

        ex). z_dim = 2, length = 3, a = [0, 1], b = [1, 3]
         [
             [0., 1.],
             [0.5, 2.],
             [1.0, 3.]
         ]
         
        return: torch.Tensor: shape = (length, z_dim)
        """
        shape = (length, z_dim)
        np.random.seed(71)
        a = np.random.uniform(-1, 1, size=z_dim)
        b = np.random.uniform(-1, 1, size=z_dim)
        z = np.zeros(shape)

        # a から b の値に徐々に変化させるため内分点を計算する
        for i in range(length):
            z[i, :] = a * i / length + b * (length - i) / length

        z = torch.from_numpy(z).type(torch.FloatTensor).to(self.device)
        return z

    def generate(self, generator, z):
        """
        :param Generator generator:
        :return:
        :rtype: torch.Tensor
        """
        with torch.no_grad():
            generator.eval()
            x = generator(z).cpu()
        return x
