import torch
from .mdm import MDM


class CLD(MDM):
    def __init__(self, config):
        self.cld_gamma = 2.0 * torch.tensor([config.stationary_aux_var]).sqrt().item()

        q = torch.tensor([[0.0, -1.0], [1.0, 0.0]])
        d = torch.tensor([[0.0, 0.0], [0.0, self.cld_gamma]])

        super().__init__(config, q_fixed=q, d_fixed=d)

        assert not self.config.d_full, "set d_full to false for cld"
