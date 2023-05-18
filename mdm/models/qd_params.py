import torch
from torch import nn


def rand_weight(shape1, shape2):
    return nn.Linear(shape1, shape2, bias=False).weight.clone()


class D_act(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.act = nn.Softplus()

    def forward(self, x):
        return self.act(x) + 1e-3


class DModule(nn.Module):
    def __init__(self, n_vars, device, d_fixed=None, full=False):
        super().__init__()
        self.full = full
        self.fixed = True if d_fixed is not None else False
        self.learned = not self.fixed
        print("Making D Module. Full {}. Fixed {}".format(self.full, self.fixed))
        
        if self.fixed:
            #d = d_fixed.to(device)
            print("using d fixed")
            d = d_fixed
        else:
            if self.full:
                d = rand_weight(n_vars, n_vars)
            else:
                d = rand_weight(n_vars, 1).squeeze(0)

        self.d = nn.Parameter(d, requires_grad=not self.fixed)
        self.act = D_act()

        if self.learned and self.full:
            assert self.D_is_psd(), "initial D not PSD"

    def get_fixed_D(
        self,
    ):
        return self.d

    def ddt(
        self,
    ):
        # unconstrained -> PSD
        d = self.d
        ddt = d @ d.permute(1, 0)
        return self.d @ self.d.permute(1, 0)

    def learned_full(
        self,
    ):
        return self.act(self.ddt())

    def learned_nonfull(
        self,
    ):
        return torch.diag(self.act(self.d))

    def get_current_D(
        self,
    ):
        if self.fixed:
            D = self.get_fixed_D()
        else:
            D = self.learned_full() if self.full else self.learned_nonfull()
        return D

    def D_is_psd(self):
        D = self.get_current_D()
        evals, evecs = torch.linalg.eig(D)
        evals_real = evals.real

        all_noneg = (evals_real >= 0).all()
        return all_noneg

    def forward(self):
        return self.get_current_D()


class QModule(nn.Module):
    def __init__(self, n_vars, device, q_fixed=None):
        super().__init__()
        self.fixed = True if q_fixed is not None else False

        #q = q_fixed.to(device) if self.fixed else rand_weight(n_vars, n_vars)
        if self.fixed:
            #q = q_fixed.to(device)
            q = q_fixed
            print("using q fixed")
        else:
            q = rand_weight(n_vars, n_vars)


        self.q = nn.Parameter(q, requires_grad=not self.fixed)
        print("Making Q Module. Fixed {}".format(self.fixed))

    def q_minus_qt(self):
        q = self.q
        qmqt = q - q.t()
        return qmqt

    def forward(self):
        Q = self.q if self.fixed else self.q_minus_qt()
        return Q
