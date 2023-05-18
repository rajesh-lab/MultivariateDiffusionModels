import math
import torch
from typing import List

from .sde_lib import SDELib
from .qd_params import QModule, DModule

from mdm.utils import batch_matrix_utils
from mdm.utils.utils import mat_square_root


class MDM(SDELib):
    def __init__(self, config, q_fixed=None, d_fixed=None):
        super().__init__(config)
        # self.save_hyperparameters()

        self.get_q_and_d_params(q_fixed=q_fixed, d_fixed=d_fixed)

        self.save_hyperparameters(ignore=["q_fixed", "d_fixed"])
        self.save_hyperparameters(ignore=["config/q_fixed", "config/d_fixed"])

        self.get_prior_dist()
        self.get_v0_dist()
        self.get_grad_H_mat()

    def monitor_params_and_grads(
        self,
    ):
        q = self.qmodule.q
        d = self.dmodule.d
        if q.grad is not None:
            qnorm = self.qmodule.q.grad.detach().data.norm(2).pow(0.5)
            self.log("grad norm q", qnorm)
        if d.grad is not None:
            dnorm = self.dmodule.d.grad.detach().data.norm(2).pow(0.5)
            self.log("grad norm d", dnorm)

        Q = self.get_Q()
        D = self.get_D()
        Q01 = Q[0, 1]
        D00 = D[0, 0]
        D11 = D[1, 1]
        self.log("Q 01", Q01.detach())
        self.log("D 00", D00.detach())
        self.log("D 11", D11.detach())

    def get_q_and_d_params(self, q_fixed, d_fixed):
        self.qmodule = QModule(self.n_vars, self.device, q_fixed=q_fixed)
        self.dmodule = DModule(
            self.n_vars, self.device, d_fixed=d_fixed, full=self.config.d_full
        )

    def get_Q(
        self,
    ):
        Q = self.qmodule()
        if self.qmodule.fixed:
            Q = Q.detach()
        return Q

    def get_D(
        self,
    ):
        D = self.dmodule()
        if self.dmodule.fixed:
            D = D.detach()
        return D

    # H(u) is scalar
    # nabla H(u) is a vecotr
    # but we represent as diagonal matrix
    # to pre-multiply it into drift matrix A
    def get_grad_H_mat(self):
        # for uncorrelated gaussian stationary,
        # grad H matrix is diagonal with inverses of stationary variances
        inv_var = torch.ones(self.config.n_vars)
        inv_var[0] *= 1.0 / self.config.stationary_x_var
        inv_var[1:] *= 1.0 / self.config.stationary_aux_var

        self.grad_H_mat = torch.diag(inv_var).to(self.device)

    def get_v0_dist_like(self, tensor):
        assert self.config.n_vars > 1
        num_total_aux = self.config.n_vars - 1
        v0_var = self.config.init_aux_var * torch.ones(num_total_aux).type_as(tensor)
        mu = torch.zeros(num_total_aux).type_as(tensor)
        q0 = torch.distributions.Normal(loc=mu, scale=v0_var.sqrt())
        return q0

    def get_v0_dist(self):
        if self.config.n_vars > 1:
            num_total_aux = self.config.n_vars - 1
            v0_var = self.config.init_aux_var * torch.ones(
                num_total_aux, device=self.device
            )
            self.v0_dist = torch.distributions.normal.Normal(
                loc=torch.zeros(num_total_aux, device=self.device),
                scale=(v0_var**0.5).to(self.device),
            )

    def get_prior_dist(self):
        stat_cov = torch.ones(self.config.n_vars)
        stat_cov[:1] *= self.config.stationary_x_var
        stat_cov[1:] *= self.config.stationary_aux_var

        self.prior_dist = torch.distributions.normal.Normal(
            loc=torch.zeros(self.config.n_vars, device=self.device),
            scale=(stat_cov**0.5).to(self.device),
        )

    def on_fit_start(self) -> None:
        super().on_fit_start()

        self.get_prior_dist()
        self.get_v0_dist()
        self.get_grad_H_mat()

    def sample_v0(self, batch_size):
        tensor = torch.ones(batch_size, device=self.device)
        qv0 = self.get_v0_dist_like(tensor)
        samples = qv0.sample((batch_size[0], self.data_dim))
        samples = samples.permute(0, 2, 1)
        samples = batch_matrix_utils.pack_vars(samples, n_vars=self.config.n_vars - 1)
        return samples

    def logq_v0(self, v0):
        unpacked_v0 = batch_matrix_utils.unpack_vars(
            v0, n_vars=self.config.n_vars - 1
        ).permute(0, 2, 1)
        qv0 = self.get_v0_dist_like(v0)
        return qv0.log_prob(unpacked_v0).sum(-1).sum(-1)

    def entropy_v0(
        self,
    ):
        # TODO ASSUMES GAUSSIAN TODO
        two_pi = 2.0 * torch.tensor([math.pi]).to(self.device)
        return 0.5 * (two_pi * self.config.stationary_aux_var).log() + 0.5

    def Qt(self, t):
        Q = self.get_Q()

        t_arr = self.expand_t(t, 3)
        beta_t = self.beta_fn(t_arr)

        return Q * beta_t

    def Dt(self, t):
        D = self.get_D()

        t_arr = self.expand_t(t, 3)
        beta_t = self.beta_fn(t_arr)

        return D * beta_t

    def int_Qt(self, t):
        Q = self.get_Q()

        t_arr = self.expand_t(t, 3)
        int_beta_t = self.int_beta_fn(t_arr)

        return Q * int_beta_t

    def int_Dt(self, t):
        D = self.get_D()

        t_arr = self.expand_t(t, 3)
        int_beta_t = self.int_beta_fn(t_arr)

        return D * int_beta_t

    def A(self, t_arr):
        A = -1 * (self.Dt(t_arr) + self.Qt(t_arr))
        A = A @ self.grad_H_mat.type_as(A)

        return A

    def f(self, u: torch.Tensor, t_arr: torch.Tensor) -> torch.Tensor:
        assert len(u.shape) == 2, "drift fn takes variable of dim 2"
        batch_size = u.shape[0]

        A = self.A(t_arr)

        assert A.shape == (batch_size, self.config.n_vars, self.config.n_vars), print(
            "A shape", A.shape
        )
        assert u.shape == (batch_size, self.total_dim), print("u shape", u.shape)

        return batch_matrix_utils.block_mat_multiply(A, u, n_vars=self.config.n_vars)

    def G(self, t_arr: torch.Tensor) -> torch.Tensor:
        if self.config.d_full:
            return mat_square_root(2 * self.Dt(t_arr))
        else:
            return torch.sqrt(2 * self.Dt(t_arr))

    def int_GGT(self, t_arr: torch.Tensor) -> torch.Tensor:
        return 2 * self.int_Dt(t_arr)

    def transition_mean_coefficient(self, t_arr: torch.Tensor) -> torch.Tensor:
        int_A = self.int_A(t_arr)
        exp_A = torch.matrix_exp(int_A)
        batch_size = t_arr.shape[0]
        assert exp_A.shape == (batch_size, self.config.n_vars, self.config.n_vars)

        return exp_A

    def transition_mean(
        self, u: torch.Tensor, t_arr: torch.Tensor, hybrid
    ) -> torch.Tensor:
        batch_size = u.shape[0]
        assert u.shape == (batch_size, self.total_dim)

        exp_A = self.transition_mean_coefficient(t_arr)

        if hybrid:
            # condition only on x, replace v0 value with its mean
            u[:, self.data_dim :] = u[:, self.data_dim :] * 0

        mean = batch_matrix_utils.block_mat_multiply(
            exp_A, u, n_vars=self.config.n_vars
        )
        return mean, exp_A

    def get_sigma_0(self, hybrid, var0_x: float = None) -> torch.Tensor:
        sigma_0 = torch.eye(self.config.n_vars, device=self.device)
        sigma_0 *= self.config.init_aux_var

        # define Sigma_0
        # condition on only x by setting its init cov to 0
        sigma_0[0, 0] = var0_x if var0_x is not None else 0
        if hybrid is False:
            # condition on both x,v0 by setting whole sigma to 0
            sigma_0 = torch.zeros_like(sigma_0)
        return sigma_0

    def GGT(self, t_arr):
        return 2 * self.Dt(t_arr)

    def int_A(self, t_arr):
        int_A = -1 * (self.int_Qt(t_arr) + self.int_Dt(t_arr))
        int_A = int_A @ self.grad_H_mat.type_as(int_A)
        return int_A

    def transition_var(self, t_arr: torch.Tensor, hybrid) -> torch.Tensor:
        int_A = self.int_A(t_arr)
        int_GGT = self.int_GGT(t_arr)

        # define the matrix ([[A, vol_volT], [0, -A.T]])
        sigma_exp_upper = torch.cat([int_A, int_GGT], dim=2)
        sigma_exp_lower = torch.cat(
            [torch.zeros_like(int_A), -int_A.permute(0, 2, 1)], dim=2
        )

        # compute exp(int_0^t beta(s) ds * [[A, vol_volT], [0, -A.T]])
        sigma_exp = torch.matrix_exp(
            torch.cat([sigma_exp_upper, sigma_exp_lower], dim=1)
        )

        # CD_0
        sigma_0 = self.get_sigma_0(hybrid=hybrid)
        CD_0 = torch.cat(
            [sigma_0, torch.eye(self.config.n_vars, device=sigma_0.device)]
        )
        CD_t = sigma_exp @ CD_0

        C_t = CD_t[:, : self.config.n_vars, :]
        assert int_A.shape == (t_arr.shape[0], self.config.n_vars, self.config.n_vars)
        D_t_inv = torch.matrix_exp(int_A.permute(0, 2, 1))

        sigma_t = torch.bmm(C_t, D_t_inv)

        return sigma_t

    def sample_from_transition_kernel(
        self,
        u_0: torch.Tensor,
        t_arr: torch.Tensor,
        hybrid,
    ) -> List[torch.Tensor]:
        mean, _ = self.transition_mean(u_0, t_arr, hybrid=hybrid)

        cov = self.transition_var(t_arr, hybrid=hybrid)
        L = batch_matrix_utils.safe_cholesky(cov, self).float()
        LT_inv = torch.linalg.inv(L.permute(0, 2, 1))

        eps = torch.randn_like(u_0, requires_grad=False)

        # u_t = mean(t) + L_t @ eps
        perturbed_data = mean + batch_matrix_utils.block_mat_multiply(
            L, eps, n_vars=self.config.n_vars
        )
        return perturbed_data, eps, LT_inv

    def div_f(self, u, t_arr):
        assert len(u.shape) == 2, "div f takes variable of dim 2"
        bsz = u.shape[0]
        At = self.A(t_arr)
        tr = batch_matrix_utils.trace_fn(At)
        assert tr.shape == (bsz,)
        data_dim = u.shape[1] / self.config.n_vars
        return tr * data_dim

    def transition_LT_inv(self, t_arr: torch.Tensor, hybrid) -> torch.Tensor:
        cov = self.transition_var(t_arr, hybrid)
        L = batch_matrix_utils.safe_cholesky(cov, self).float()
        LT_inv = torch.linalg.inv(L.permute(0, 2, 1))
        return LT_inv
