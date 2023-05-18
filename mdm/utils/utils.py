import os
import math
import argparse
import yaml
import torch
import torchvision
import matplotlib.pyplot as plt
import math
import numpy as np
from torch import Tensor
import torch.nn as nn
from torch import optim
from torch.autograd import Function
import scipy.linalg


def print_config_nicely(config):
    for key in sorted(vars(config)):
        print(key, getattr(config, key))


def batch_transpose(A):
    return A.permute(0, 2, 1)


def inv(A):
    return torch.linalg.inv(A)


def randn_like(u):
    return torch.randn_like(u)


def grad(y, x, v):
    return torch.autograd.grad(y, x, v)[0]


def vjp(f, x, v):
    return torch.autograd.functional.vjp(f, x, v=v)


def dot(a, b):
    return (a * b).sum(-1)


def show_imgs(imgs, title=None, row_size=4, save_img=False, img_filename=None):
    # Form a grid of pictures (we use max. 8 columns)
    num_imgs = imgs.shape[0] if isinstance(imgs, Tensor) else len(imgs)
    is_int = (
        imgs.dtype == torch.int32
        if isinstance(imgs, Tensor)
        else imgs[0].dtype == torch.int32
    )
    nrow = min(num_imgs, row_size)
    ncol = int(math.ceil(num_imgs / nrow))
    imgs = torchvision.utils.make_grid(
        imgs, nrow=nrow, pad_value=128 if is_int else 0.5
    )

    np_imgs = imgs.cpu().numpy()

    # Plot the grid
    plt.figure(figsize=(1.5 * nrow, 1.5 * ncol))
    plt.imshow(np.transpose(np_imgs, (1, 2, 0)), interpolation="nearest")

    plt.axis("off")
    if title is not None:
        plt.title(title)

    if save_img:
        plt.savefig(img_filename)
        plt.show()
        plt.close()

        return np_imgs

    plt.show()
    plt.close()


def get_config(config_path):
    if config_path is not None:
        with open(config_path, mode="r") as fp:
            sde_config = yaml.safe_load(fp)

    config = argparse.Namespace(**sde_config)
    return config


class CosineWarmupScheduler(optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup, max_iters):
        self.warmup = warmup
        self.max_num_iters = max_iters
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.max_num_iters))
        if epoch <= self.warmup:
            lr_factor *= epoch * 1.0 / self.warmup
        return lr_factor


# noinspection PyUnusedLocal
class CenterTransform(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)

    def forward_transform(self, x, logpx=None):
        # Rescale from [0, 1] to [-1, 1]
        y = x * 2.0 - 1.0
        if logpx is None:
            return y
        return y, logpx + self._logdetgrad(x).view(x.size(0), -1).sum(1)

    def reverse(self, y, logpy=None, **kwargs):
        # Rescale from [-1, 1] to [0, 1]
        x = (y + 1.0) / 2.0
        if logpy is None:
            return x
        return x, logpy - self._logdetgrad(x).view(x.size(0), -1).sum(1)

    def _logdetgrad(self, x):
        return (torch.ones_like(x) * 2).log()

    def __repr__(self):
        return "{name}({alpha})".format(name=self.__class__.__name__, **self.__dict__)


# noinspection PyUnusedLocal
class LogitTransform(nn.Module):
    """
    The proprocessing step used in Real NVP:
    y = sigmoid(x) - a / (1 - 2a)
    x = logit(a + (1 - 2a)*y)
    """

    def __init__(self, alpha):
        nn.Module.__init__(self)
        self.alpha = alpha

    def forward_transform(self, x, logpx=None):
        s = self.alpha + (1 - 2 * self.alpha) * x
        y = torch.log(s) - torch.log(1 - s)
        if logpx is None:
            return y
        return y, logpx + self._logdetgrad(x).view(x.size(0), -1).sum(1)

    def reverse(self, y, logpy=None, **kwargs):
        x = (torch.sigmoid(y) - self.alpha) / (1 - 2 * self.alpha)
        if logpy is None:
            return x
        return x, logpy - self._logdetgrad(x).view(x.size(0), -1).sum(1)

    def _logdetgrad(self, x):
        s = self.alpha + (1 - 2 * self.alpha) * x
        logdetgrad = -torch.log(s - s * s) + math.log(1 - 2 * self.alpha)
        return logdetgrad

    def __repr__(self):
        return "{name}({alpha})".format(name=self.__class__.__name__, **self.__dict__)


# from https://github.com/steveli/pytorch-sqrtm
class MatrixSquareRoot(Function):
    """Square root of a positive definite matrix.
    NOTE: matrix square root is not differentiable for matrices with
          zero eigenvalues.
    """

    @staticmethod
    def forward(ctx, input):
        m = input.detach().cpu().numpy().astype(np.float_)
        sqrtm = torch.from_numpy(scipy.linalg.sqrtm(m).real).to(input)
        ctx.save_for_backward(sqrtm)
        return sqrtm

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = None
        if ctx.needs_input_grad[0]:
            (sqrtm,) = ctx.saved_tensors
            sqrtm = sqrtm.data.cpu().numpy().astype(np.float_)
            gm = grad_output.data.cpu().numpy().astype(np.float_)

            # Given a positive semi-definite matrix X,
            # since X = X^{1/2}X^{1/2}, we can compute the gradient of the
            # matrix square root dX^{1/2} by solving the Sylvester equation:
            # dX = (d(X^{1/2})X^{1/2} + X^{1/2}(dX^{1/2}).
            grad_sqrtm = scipy.linalg.solve_sylvester(sqrtm, sqrtm, gm)

            grad_input = torch.from_numpy(grad_sqrtm).to(grad_output)
        return grad_input


def mat_square_root(A):
    # sqrtm = MatrixSquareRoot.apply
    # return sqrtm(A)

    U, S, V = torch.linalg.svd(A, full_matrices=False)
    s = torch.sqrt(S)
    rootA = U @ torch.diag_embed(s) @ V
    return rootA
