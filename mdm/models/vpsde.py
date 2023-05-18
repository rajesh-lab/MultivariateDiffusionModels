import torch
from torch import nn
from typing import List

from mdm.models import SDELib


class VPSDE(SDELib):
    def __init__(self, config):
        super().__init__(config)
        self.get_prior_dist()

    def get_prior_dist(self):
        self.prior_dist = torch.distributions.normal.Normal(
            loc=nn.Parameter(
                torch.zeros(self.n_vars, device=self.device), requires_grad=False
            ),
            scale=nn.Parameter(
                torch.ones(self.n_vars, device=self.device) ** 0.5, requires_grad=False
            ),
        )

    def on_fit_start(self) -> None:
        super().on_fit_start()
        self.get_prior_dist()

    def monitor_params_and_grads(
        self,
    ):
        pass

    def f(self, u: torch.Tensor, t_arr: torch.Tensor) -> torch.Tensor:
        t_arr = self.expand_t(t_arr, 2)
        beta_t = self.beta_fn(t_arr)
        return -0.5 * beta_t * u

    def G(self, t_arr: torch.Tensor) -> torch.Tensor:
        t_arr = self.expand_t(t_arr, 2)
        beta_t = self.beta_fn(t_arr)
        return torch.sqrt(beta_t).unsqueeze(-1)

    def GGT(self, t_arr: torch.Tensor) -> torch.Tensor:
        t_arr = self.expand_t(t_arr, 3)
        return self.beta_fn(t_arr)

    def div_f(self, u: torch.Tensor, t_arr: torch.Tensor) -> torch.Tensor:
        return -0.5 * self.beta_fn(t_arr) * u.shape[1]

    def transition_mean(self, u: torch.Tensor, t_arr: torch.Tensor, hybrid) -> torch.Tensor:
        coef = self.transition_mean_coefficient(t_arr)
        mean = coef * u
        return mean, coef

    def transition_mean_coefficient(self, t_arr):
        t_arr = self.expand_t(t_arr, 2)
        beta_max = self.config.beta_1
        beta_min = self.config.beta_0

        int_beta_t = self.int_beta_fn(t_arr)

        coef = torch.exp(-0.5 * int_beta_t)
        return coef

    def transition_var(self, t_arr: torch.Tensor, hybrid) -> torch.Tensor:
        t_arr = self.expand_t(t_arr, 2)
        int_beta_t = self.int_beta_fn(t_arr)

        return 1 - torch.exp(-int_beta_t)

    def transition_std(self, t_arr, hybrid):
        return torch.sqrt(self.transition_var(t_arr, hybrid))

    def sample_from_transition_kernel(
        self,
        u_0: torch.Tensor,
        t_arr: torch.Tensor,
        hybrid,
    ) -> List[torch.Tensor]:

        batch_size = u_0.shape[0]
        u_0 = u_0.view(batch_size, -1)

        mean, _ = self.transition_mean(u=u_0, t_arr=t_arr, hybrid=hybrid)
        std = torch.sqrt(self.transition_var(t_arr=t_arr, hybrid=hybrid)).to(u_0.device)

        eps = torch.randn_like(u_0)

        perturbed_data = mean + eps * std
        LT_inv = 1.0 / std

        for tensor in [u_0, mean, eps, perturbed_data]:
            assert tensor.shape == (batch_size, self.total_dim)

        return perturbed_data, eps, LT_inv.unsqueeze(-1)

    def transition_LT_inv(self, t_arr: torch.Tensor, hybrid) -> torch.Tensor:
        std = torch.sqrt(self.transition_var(t_arr=t_arr, hybrid=hybrid))
        return (1.0 / std).unsqueeze(-1)
