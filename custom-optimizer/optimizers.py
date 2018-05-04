# coding: utf-8
"""
自前定義の最適化アルゴリズムの定義
"""
import torch
from torch.optim import Optimizer

__author__ = "nyk510"


class RDA(Optimizer):
    """
    Implements Regularized Dual Averaging method.

    This implementation is enhanced l1-RDA method described in
    https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/xiao10JMLR.pdf
    """

    def __init__(self, params, gamma=5e4, rho=1e-4, lam=0.1):
        defaults = dict(gamma=gamma, rho=rho, lam=lam)
        super(RDA, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad.data

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["dual_average"] = torch.zeros_like(p.data)

                state["step"] += 1

                dual_average = state["dual_average"]
                step = state["step"]
                gamma, lam, rho = group["gamma"], group["lam"], group["rho"]

                dual_average.mul_(step - 1).add_(grad).div_(step)

                lambda_t = lam + gamma * rho / (step ** .5)
                p.data = - (step ** .5) / gamma * (dual_average - lambda_t * dual_average.sign())
                p.data[dual_average.abs() <= lambda_t] = 0

        return loss


class L1SGD(Optimizer):
    """
    Stochastic Gradient Descent with l1 normalization
    """

    def __init__(self, params, total, gamma=1e2, lam=.1):
        """

        :param iterable params:
        :param float gamma:
        :param int total: total size of dataset
        :param float lam: l1 regularization parameter
        """
        stepsize = 1. / gamma * ((2. / total) ** .5)
        defaults = dict(gamma=gamma, stepsize=stepsize, lam=lam)
        super(L1SGD, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad.data

                stepsize, lam = group["stepsize"], group["lam"]
                diff = grad.add(p.data.sign().mul(lam)).mul(stepsize)
                p.data.sub_(diff)
        return loss
