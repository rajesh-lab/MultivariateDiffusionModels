import torch

from .mdm import MDM


class MALDA(MDM):
    def __init__(self, config):
        K = config.n_vars
        ones = torch.ones(K, K)
        upper = -1.0 * ones.triu(diagonal=1)
        lower = 1.0 * ones.tril(diagonal=-1)
        Q = (upper + lower) / 1.0
        assert torch.equal(-Q, Q.permute(1, 0))

        M = config.stationary_aux_var
        Gamma = 2.0 * torch.tensor([M]).sqrt().item()
        Gamma = torch.ones(K) * Gamma
        Gamma[0] = 0
        D = torch.diag(Gamma)

        super().__init__(config, q_fixed=Q, d_fixed=D)
