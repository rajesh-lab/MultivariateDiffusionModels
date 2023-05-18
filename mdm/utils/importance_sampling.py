import torch
from torch import optim
import numpy as np


class VPImpSamp:
    def __init__(self, sde, beta_0, beta_1, t_min, t_max):
        self.sde = sde

        self.beta_0 = torch.tensor([beta_0])
        self.beta_1 = torch.tensor([beta_1])

        self.t_max = torch.tensor([t_max])
        self.t_min = torch.tensor([t_min])

    def sample(self, n_samples):
        t_samples = self.sample_importance_weighted_time_for_likelihood(
            shape=n_samples, eps=self.t_min
        )
        return t_samples.clamp(min=self.t_min, max=self.t_max)

    def likelihood_importance_cum_weight(self, t, eps):
        exponent1 = 0.5 * eps * (eps - 2) * self.beta_0 - 0.5 * eps**2 * self.beta_1
        exponent2 = 0.5 * t * (t - 2) * self.beta_0 - 0.5 * t**2 * self.beta_1
        term1 = torch.where(
            torch.abs(exponent1) <= 1e-3, -exponent1, 1.0 - torch.exp(exponent1)
        )
        term2 = torch.where(
            torch.abs(exponent2) <= 1e-3, -exponent2, 1.0 - torch.exp(exponent2)
        )
        return 0.5 * (
            -2 * torch.log(term1)
            + 2 * torch.log(term2)
            + self.beta_0 * (-2 * eps + eps**2 - (t - 2) * t)
            + self.beta_1 * (-(eps**2) + t**2)
        )

    def sample_importance_weighted_time_for_likelihood(
        self, shape, eps, quantile=None, steps=100
    ):
        Z = self.likelihood_importance_cum_weight(self.t_max, eps=eps)
        if quantile is None:
            quantile = torch.rand(shape) * (Z - 0) + 0
        lb = torch.ones_like(quantile) * eps
        ub = torch.ones_like(quantile) * self.t_max

        def bisection_func(carry, idx):
            lb, ub = carry
            mid = (lb + ub) / 2.0
            value = self.likelihood_importance_cum_weight(mid, eps=eps)
            lb = torch.where(value <= quantile, mid, lb)
            ub = torch.where(value <= quantile, ub, mid)
            return (lb, ub), idx

        carry = (lb, ub)
        for i in range(steps):
            carry, _ = bisection_func(carry, i)
        (lb, ub) = carry
        # (lb, ub), _ = jax.lax.scan(bisection_func, (lb, ub), jnp.arange(0, steps))
        return (lb + ub) / 2.0

    def r(self, t):
        bsz = t.shape[0]

        int_beta_t = self.sde.int_beta_fn(self.sde.expand_t(t, 2))
        var = 1 - torch.exp(-int_beta_t)

        g2 = self.sde.beta_fn(t)
        assert var.shape == (bsz, 1)

        var = var.squeeze(-1)
        ratio = g2 / var

        assert ratio.shape == (bsz,)
        return ratio

    def Z(
        self,
    ):
        return self.likelihood_importance_cum_weight(t=self.t_max, eps=self.t_min)

    def weight(self, t):
        return self.r(t) / self.Z().to(t)
