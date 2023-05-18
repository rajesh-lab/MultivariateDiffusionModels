from .qd_params import QModule, DModule
from .mdm import MDM


class LearnedSDE(MDM):
    def __init__(self, config, q_fixed=None, d_fixed=None):
        super().__init__(config)

        self.get_q_and_d_params(q_fixed=q_fixed, d_fixed=d_fixed)

        self.get_prior_dist()
        self.get_v0_dist()
        self.get_grad_H_mat()
