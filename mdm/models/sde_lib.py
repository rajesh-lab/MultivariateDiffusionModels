import pytorch_lightning as pl
import torch
from torch import optim
import numpy as np
from tqdm.auto import tqdm
from pytorch_lightning.utilities import rank_zero_only

from torchdiffeq import odeint

from mdm.score_models import TinyUNet, MLP, HoUNet
from mdm.bin.evaluate import train_loss_fn, nelbo
from mdm.utils.batch_matrix_utils import (
    block_mat_multiply,
    unpack_vars,
    pack_vars,
)
from mdm.utils.utils import (
    LogitTransform,
    CenterTransform,
    CosineWarmupScheduler,
)
from mdm.utils.importance_sampling import VPImpSamp


class SDELib(pl.LightningModule):
    def __init__(self, config):
        super().__init__()

        self.config = config

        # validate config args
        assert self.config.beta_1 >= self.config.beta_0

        # get var dimensions
        self.n_vars = self.config.n_vars
        if self.config.is_image:
            self.data_dim = (
                self.config.in_channels * self.config.height * self.config.width
            )
        else:
            self.data_dim = self.config.dim

        self.total_dim = self.n_vars * self.data_dim

        print(f"Using {self.config.transform}")
        if self.config.transform == "logit":
            self.transform = LogitTransform(alpha=0.05)
        elif self.config.transform == "center":
            self.transform = CenterTransform()
        else:
            raise NotImplementedError(
                f"Transform {self.config.transform} not implemented"
            )

        # load model
        self.get_model()

        self.ism_repeats = 2

        self.val_metric_keys = [
            "val_loss",
            "val_ism_bpd",
            "val_weighted_dsm_bpd_offset",
            "val_weighted_dsm_bpd_no_offset",
            "val_unweighted_dsm_bpd",
        ]

        self.test_metric_keys = [
            "test_loss",
            "test_ism_bpd",
            "test_weighted_dsm_bpd_offset",
            "test_weighted_dsm_bpd_no_offset",
            "test_unweighted_dsm_bpd",
        ]

        self.best_val_metrics = {
            "val_loss": np.inf,
            "val_ism_bpd": np.inf,
            "val_weighted_dsm_bpd_offset": np.inf,
            "val_weighted_dsm_bpd_no_offset": np.inf,
            "val_unweighted_dsm_bpd": np.inf,
        }

        # self.save_hyperparameters()
        self.save_hyperparameters(ignore=["q_fixed", "d_fixed"])
        self.save_hyperparameters(ignore=["config/q_fixed", "config/d_fixed"])

        self.eval_step_outputs = dict(test=[], val=[])

    def monitor_params_and_grads(
        self,
    ):
        raise NotImplementedError

    def get_model(
        self,
    ):
        # load model
        if self.config.score_model_type == "TinyUNet":
            # NOTE DOES NOT TAKE DROP OUT
            self.deep_model = TinyUNet(
                in_channels=self.config.in_channels,
                out_channels=self.config.out_channels,
                height=self.config.height,
                width=self.config.width,
                n_vars=self.config.n_vars,
            )
        elif self.config.score_model_type == "MLP":
            self.deep_model = MLP(input_dim=self.total_dim, out_dim=self.total_dim)
        elif self.config.score_model_type == "HoUNet":
            if self.config.dataset in ["cifar", "imagenet"]:
                self.deep_model = HoUNet(
                    config=self.config,
                    ch=128,
                    ch_mult=(1, 2, 2, 2),
                )
            elif self.config.dataset == "mnist":
                self.deep_model = HoUNet(
                    config=self.config,
                    ch=32,
                    ch_mult=(1, 2, 2),
                )
            else:
                raise NotImplementedError(
                    f"dataset  {self.config.dataset} not implemented"
                )
        elif self.config.score_model_type == "ncsnpp":
            from mdm.score_models.biggg_model.models.ncsnpp import (
                NCSNpp,
            )

            self.deep_model = NCSNpp(self.config)
        else:
            raise NotImplementedError(
                f"Score-model type {self.config.score_model_type} not implemented"
            )

    def beta_fn(self, t):
        """
        beta(t) function. hom return beta_0, inhom beta_0 + t (beta_1 - beta_0)
        """
        if self.config.beta_fn_type == "inhom":
            return self.config.beta_0 + t * (self.config.beta_1 - self.config.beta_0)
        elif self.config.beta_fn_type == "hom":
            return self.config.beta_0 * torch.ones_like(t)
        else:
            raise NotImplementedError(
                f"Beta fn type {self.config.beta_fn_type} not implemented"
            )

    def int_beta_fn(self, t):
        """
        integral of beta(t) function. hom return t * beta_0, inhom t * beta_0 + t**2/2 (beta_1 - beta_0)
        """
        if self.config.beta_fn_type == "inhom":
            return self.config.beta_0 * t + (
                self.config.beta_1 - self.config.beta_0
            ) * (t**2 / 2)
        elif self.config.beta_fn_type == "hom":
            return self.config.beta_0 * t
        else:
            raise NotImplementedError(
                f"Beta fn type {self.config.beta_fn_type} not implemented"
            )

    def _make_prior_dist_like(self, u):
        if self.n_vars == 1:
            mu = torch.zeros(1, requires_grad=False).type_as(u)
            sig = torch.ones(1, requires_grad=False).sqrt().type_as(u)
            return torch.distributions.Normal(loc=mu, scale=sig)
        else:
            stat_cov = torch.ones(self.config.n_vars, requires_grad=False)
            stat_cov[:1] *= self.config.stationary_x_var
            stat_cov[1:] *= self.config.stationary_aux_var
            return torch.distributions.Normal(
                loc=torch.zeros(self.config.n_vars, requires_grad=False).type_as(u),
                scale=stat_cov.sqrt().type_as(u),
            )

    def prior_logp(self, u: torch.Tensor) -> torch.Tensor:
        """
        compute prior probability p(u_0)
        """
        unpacked_u = unpack_vars(u, n_vars=self.config.n_vars).permute(0, 2, 1)

        pdist = self._make_prior_dist_like(u)
        return pdist.log_prob(unpacked_u).sum(-1).sum(-1)

    def sample_from_prior(self, n_samples: int) -> torch.Tensor:
        """
        sample from prior distribution u ~ p(u_0)
        """

        some_tensor = torch.ones(3).to(self.device)
        pdist = self._make_prior_dist_like(some_tensor)
        samples = pdist.sample((n_samples, self.data_dim))
        samples = samples.permute(0, 2, 1)
        samples = pack_vars(samples, n_vars=self.config.n_vars)
        return samples

    def make_u0(self, batch):
        n_vars = self.config.n_vars
        batch_size = batch.shape[0]
        assert batch.shape == (batch_size, self.data_dim)

        if n_vars == 1:
            u_0 = batch
            v_0 = []
        elif n_vars > 1:
            v_0 = self.sample_v0((batch_size,))
            u_0 = torch.cat([batch, v_0], dim=1)
        assert u_0.shape == (batch_size, self.total_dim)

        return u_0, v_0

    def f(self, u, t):
        """
        Compute the drift of the sde
        """
        raise NotImplementedError

    def G(self, t):
        """
        Compute the volatility of the sde
        """
        raise NotImplementedError

    def GGT(self, t):
        raise NotImplementedError

    def transition_mean_coefficient(self, t_arr):
        raise NotImplementedError

    def transition_mean(self, batch, t_arr, hybrid):
        raise NotImplementedError

    def transition_var(self, t_arr, hybrid):
        raise NotImplementedError

    def transition_LT_inv(self, t_arr, hybrid):
        raise NotImplementedError

    def sample_from_transition_kernel(self, u_0, t_arr, hybrid):
        raise NotImplementedError

    def compute_bpd(self, neg_elbo: torch.Tensor, ldj: torch.Tensor) -> torch.Tensor:
        """
        Computes bits-per-dim.
        bpd = neg_elbo / dim * log(2) + 8
        """
        dimx = self.data_dim
        elbo = -neg_elbo
        elbo += ldj

        bpd = -(elbo / dimx) / np.log(2) + 8
        return bpd

    def expand_t(self, t, dims):
        """
        adds extra dimensions to t
        """
        assert len(t.shape) == 1
        for _ in range(1, dims):
            t = t.unsqueeze(-1)

        return t.to(self.device)

    def batch_scaler(self, batch):
        """Data normalizer. Assume data are always in [0, 1]."""
        return self.transform.forward_transform(batch, 0)

    def inverse_batch_scaler(self, batch):
        """Inverse data normalizer."""
        return self.transform.reverse(batch)

    def uniform_dequantization(self, batch):
        return (batch * 255 + torch.rand_like(batch)) / 256

    def score_fn(self, u: torch.Tensor, t_arr: torch.Tensor) -> torch.Tensor:
        hybrid = self.config.hybrid_transition_kernel

        if self.config.score_parameterization == "noise_pred":
            eps_pred = self.deep_model(u, t_arr)
            LT_inv = self.transition_LT_inv(t_arr, hybrid=hybrid)

            score_fn_val = block_mat_multiply(
                -LT_inv, eps_pred, n_vars=self.config.n_vars
            )
            return score_fn_val
        elif self.config.score_parameterization == "score_pred":
            return self.deep_model(u, t_arr)
        else:
            raise NotImplementedError(
                f"Score param {self.config.score_parameterization} not implemented"
            )

    def eps_pred(self, u: torch.Tensor, t_arr: torch.Tensor) -> torch.Tensor:
        """Prevent mark from getting a headache"""
        hybrid = self.config.hybrid_transition_kernel
        if self.config.score_parameterization == "noise_pred":
            return self.deep_model(u, t_arr)
        elif self.config.score_parameterization == "score_pred":
            s = self.deep_model(u, t_arr)
            LT_inv = self.transition_LT_inv(t_arr, hybrid=hybrid)
            LT = torch.linalg.inv(LT_inv)

            return block_mat_multiply(-LT, s, n_vars=self.config.n_vars)
        else:
            raise NotImplementedError(
                f"Eps pred {self.config.score_parameterization} not implemented"
            )

    def process_batch(self, batch):
        batch, _ = batch
        batch_size = batch.shape[0]
        batch = batch.view(batch_size, -1)
        batch = self.uniform_dequantization(batch)
        batch, ldj = self.batch_scaler(batch)
        return batch, ldj

    def decide_loss(
        self,
    ):
        sde_type = self.config.sde_type
        loss = self.config.train_loss_type
        if loss == "noise_pred_to_elbo":
            D_00_is_0 = sde_type in ["alda", "malda", "cld"]
            noise_type = "cld_noise_pred" if D_00_is_0 else "noise_pred"

            switch = self.current_epoch >= self.config.switch_epoch
            loss = "dsm_elbo" if switch else noise_type
        return loss

    def training_step(self, batch, batch_idx):
        batch, _ = self.process_batch(batch)
        loss_type = self.decide_loss()

        train_loss = train_loss_fn(
            data=batch,
            loss_type=loss_type,
            sde=self,
        )
        assert train_loss.shape == (batch.shape[0],)
        train_loss = train_loss.mean()
        self.log("train_loss", train_loss, on_step=True, prog_bar=True, sync_dist=True)

        return {"loss": train_loss}

    def evaluate_step(self, batch, batch_idx, stage):
        batch, ldj = self.process_batch(batch)
        batch_size = batch.shape[0]
        nelbo_dict = {
            "batch": batch,
            "sde": self,
            "mode": "eval",
            "MC": self.config.elbo_mc_samples_eval,
            "fixed_time_array": None,
            "hutch_mc": None,
        }

        weighted_dsm_nelbo_dict = nelbo(
            elbo_type="dsm_elbo", importance_weight=True, **nelbo_dict
        )
        weighted_dsm_nelbo_offset = weighted_dsm_nelbo_dict["nelbo_offset"]
        weighted_dsm_nelbo_no_offset = weighted_dsm_nelbo_dict["nelbo_no_offset"]

        ism_bpd = torch.tensor([0])  # self.compute_bpd(ism_nelbo, ldj)

        weighted_dsm_bpd_offset = self.compute_bpd(weighted_dsm_nelbo_offset, ldj)
        weighted_dsm_bpd_no_offset = self.compute_bpd(weighted_dsm_nelbo_no_offset, ldj)

        unweighted_dsm_bpd = torch.tensor([0])

        loss = weighted_dsm_bpd_offset

        ret = {
            "{}_loss".format(stage): loss.sum().item(),
            "{}_ism_bpd".format(stage): ism_bpd.sum().item(),
            "{}_weighted_dsm_bpd_offset".format(
                stage
            ): weighted_dsm_bpd_offset.sum().item(),
            "{}_weighted_dsm_bpd_no_offset".format(
                stage
            ): weighted_dsm_bpd_no_offset.sum().item(),
            "{}_unweighted_dsm_bpd".format(stage): unweighted_dsm_bpd.sum().item(),
            "{}_batch_size".format(stage): batch_size,
        }

        return ret

    def validation_step(self, batch, batch_idx):
        ret = self.evaluate_step(batch, batch_idx, stage="val")
        self.eval_step_outputs["val"].append(ret)

    def test_step(self, batch, batch_idx):
        ret = self.evaluate_step(batch, batch_idx, stage="test")
        self.eval_step_outputs["test"].append(ret)

    def update_best_val_metrics(self, metrics):
        for key in self.best_val_metrics:
            if key not in metrics:
                continue
            self.best_val_metrics[key] = min(metrics[key], self.best_val_metrics[key])

    def val_or_test_epoch_end(self, stage):
        outputs = self.eval_step_outputs[stage]

        keys = self.val_metric_keys if stage == "val" else self.test_metric_keys
        metrics = {key: 0.0 for key in keys}

        N = 0
        for logs in outputs:
            for key in keys:
                metrics[key] += logs[key]
            N += logs["{}_batch_size".format(stage)]

        for key in keys:
            metrics[key] /= N

        if stage == "val":
            self.update_best_val_metrics(metrics)

        for key in keys:
            self.log(
                key, metrics[key], prog_bar=True, on_epoch=True, rank_zero_only=True
            )

            if stage == "val" and key in self.best_val_metrics:
                self.log(
                    "best_" + key,
                    self.best_val_metrics[key],
                    prog_bar=True,
                    on_epoch=True,
                    rank_zero_only=True,
                )

        self.eval_step_outputs[stage].clear()

    def on_validation_epoch_end(self):
        self.val_or_test_epoch_end(stage="val")

    def on_test_epoch_end(self):
        self.val_or_test_epoch_end(stage="test")

    def configure_optimizers(self):
        if self.config.optim_type == "adam":
            optimizer = optim.Adam(
                self.parameters(),
                lr=self.config.lr,
                weight_decay=self.config.weight_decay,
            )
        elif self.config.optim_type == "adamw":
            optimizer = optim.AdamW(
                self.parameters(),
                lr=self.config.lr,
                weight_decay=self.config.weight_decay,
            )
        elif self.config.optim_type == "adamax":
            optimizer = optim.Adamax(
                self.parameters(),
                lr=self.config.lr,
                weight_decay=self.config.weight_decay,
            )
        else:
            raise NotImplementedError(f"Optim {self.config.optim_type} not implemented")

        if self.config.lr_scheduling:
            print("Using LR scheduling")

            lr_scheduler = CosineWarmupScheduler(
                optimizer=optimizer,
                warmup=self.config.warmup_iters,
                max_iters=self.config.lr_sched_max_iters,
            )

            return [optimizer], [{"scheduler": lr_scheduler, "interval": "step"}]
        else:
            return [optimizer]

    def reshape_sample(self, samples):
        batch_size = samples.shape[0]
        if self.config.is_image:
            return samples.view(
                batch_size,
                self.config.in_channels,
                self.config.width,
                self.config.height,
            )
        else:
            return samples

    def offset_likelihood(self, u_eps, eps_vec):
        mean_coef_eps = self.transition_mean_coefficient(eps_vec)

        # this thing might be tiny, esp for alda
        var_eps = self.transition_var(eps_vec, hybrid=False)

        if self.config.sde_type == "vpsde":
            var_eps = var_eps.unsqueeze(2)
            mean_coef_eps = mean_coef_eps.unsqueeze(2)

        inv_mean_coef = torch.linalg.inv(mean_coef_eps)
        ll_mean_coef = torch.bmm(inv_mean_coef, var_eps)

        hybrid = self.config.hybrid_transition_kernel
        s = self.score_fn(u_eps, eps_vec)

        ll_mean_1 = block_mat_multiply(ll_mean_coef, s, n_vars=self.config.n_vars)
        ll_mean_2 = block_mat_multiply(inv_mean_coef, u_eps, n_vars=self.config.n_vars)
        ll_mean = ll_mean_1 + ll_mean_2

        ll_cov = torch.bmm(
            inv_mean_coef, torch.bmm(var_eps, inv_mean_coef.permute(0, 2, 1))
        )

        return ll_mean, ll_cov

    def generate_samples(self, n_samples, steps=None):
        if steps is None:
            steps = self.config.n_FEs

        if self.config.sampling_method == "EM":
            x = self.EM(n_samples, steps, jump=False)
            return x
        else:
            raise NotImplementedError(
                f"Sampling Methods {self.config.sampling_method} not implemented"
            )

    def generate_sample_batches(self, n_samples, steps=None, sample_batch_size=128):
        n_samples_batches = (n_samples + sample_batch_size) // sample_batch_size

        sample_arr = []
        for _ in tqdm(range(n_samples_batches)):
            samples = self.generate_samples(n_samples=sample_batch_size)
            sample_arr.append(samples)
        sample_arr = torch.cat(sample_arr, dim=0)

        return sample_arr

    def get_sampling_t_arr(self, steps):
        t_final = self.config.T_max - self.config.T_min_sampling
        t_arr = torch.linspace(
            0,
            t_final,
            steps=1 + steps,
            device=self.device,
        )
        if self.config.sampling_t_arr_fn == "linear":
            return t_arr
        elif self.config.sampling_t_arr_fn == "quadratic":
            return t_final * torch.flip(1 - (t_arr / t_final) ** 2, dims=[0])
        else:
            raise NotImplementedError(
                f"Sampling t_arr_fn {self.config.sampling_t_arr_fn} not implemented"
            )

    def reverse_sde(self, u, t, probability_flow=False):
        hybrid = self.config.hybrid_transition_kernel

        T = self.config.T_max
        batch_size = u.shape[0]
        rev_t = T - t

        s = self.score_fn(u, rev_t)

        f = self.f(u, rev_t)
        G = self.G(rev_t)
        GG_T = self.GGT(rev_t)

        g2s = block_mat_multiply(GG_T, s, n_vars=self.config.n_vars)

        rev_drift = -f + g2s * (0.5 if probability_flow else 1.0)
        rev_diff = torch.zeros_like(G) if probability_flow else G

        assert rev_drift.shape == (batch_size, self.total_dim)
        assert rev_diff.shape == (batch_size, self.config.n_vars, self.config.n_vars)

        return rev_drift, rev_diff

    def clean_generated_samples(self, samples):
        # this implements "denoise=True" in the other repos
        samples = torch.chunk(samples, chunks=self.config.n_vars, dim=1)[0]
        samples = self.inverse_batch_scaler(samples)
        samples = self.reshape_sample(samples)
        samples = torch.clip(samples * 255, 0, 255)
        return samples

    def EM(self, n_samples, steps, jump=False):
        samples = self.sample_from_prior(n_samples)
        t_arr = self.get_sampling_t_arr(steps)
        batch_size = samples.shape[0]
        for i in range(steps):
            t = t_arr[i]
            dt = t_arr[i + 1] - t_arr[i]

            t = torch.ones(n_samples, device=self.device) * t

            f_rev, G_rev = self.reverse_sde(samples, t, probability_flow=False)
            eps = torch.randn(n_samples, self.total_dim, device=self.device)

            no_noise_samples = samples + f_rev * dt

            samples = no_noise_samples + block_mat_multiply(
                G_rev, eps, n_vars=self.config.n_vars
            ) * torch.sqrt(torch.abs(dt))

        clean = self.clean_generated_samples(no_noise_samples)

        return clean
