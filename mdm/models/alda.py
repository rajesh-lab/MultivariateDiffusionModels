import torch
from .mdm import MDM


class ALDA(MDM):
    def __init__(self, config):
        if "alda_psi" not in vars(config):
            print("Alda init'ed without psi, adding psi=1.")
            config.alda_psi = 1.0

        if "alda_gamma" not in vars(config):
            config.alda_gamma = 1.0
            print("Alda init'ed without gamma, adding gamma=1.")

        self.alda_L = 1 / config.stationary_aux_var
        self.alda_gamma = config.alda_gamma
        self.alda_psi = config.alda_psi

        q_fixed = torch.tensor(
            [
                [0, -1.0 / self.alda_L, 0],
                [1.0 / self.alda_L, 0, -self.alda_gamma],
                [0, self.alda_gamma, 0],
            ]
        )
        d_fixed = torch.tensor(
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, self.alda_psi / self.alda_L]]
        )
        super().__init__(config, q_fixed=q_fixed, d_fixed=d_fixed)
