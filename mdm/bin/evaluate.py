import torch

from mdm.utils.utils import (
    grad,
    vjp,
    dot,
    randn_like,
    batch_transpose,
)
from mdm.utils.importance_sampling import VPImpSamp
from mdm.utils.batch_matrix_utils import (
    block_mat_multiply,
    unpack_vars,
    block_quadratic_form,
    trace_fn,
)


def GGs(u, t, sde):
    n_vars = sde.config.n_vars
    GGT = sde.GGT(t)
    s = sde.score_fn(u, t)
    GGT_s = block_mat_multiply(GGT, s, n_vars=n_vars)
    return GGT_s


# one sample of hutch estimator
def hutch(f, u, method="vjp"):
    batch_size, dimu = u.shape
    v = randn_like(u)
    u.grad = None
    u.requires_grad_(True)
    assert u.shape == v.shape

    if method == "vjp":
        v_nabla = vjp(f, u, v)[1]

    elif method == "grad":
        v_nabla = grad(f(u), u, v)

    assert v_nabla.shape == (batch_size, dimu)
    v_nabla_v = dot(v_nabla, v)
    assert v_nabla_v.shape == (batch_size,)
    u.requires_grad_(False)
    return v_nabla_v


# MC estimator of div(GGs)
def compute_div_GGs(u, t, sde, MC, method="vjp"):
    assert MC is not None
    batch_size, _ = u.shape
    f = lambda x: GGs(x, t, sde)
    with torch.enable_grad():
        lst = [hutch(f, u, method) for _ in range(MC)]
    lst = torch.stack(lst, dim=1)
    assert lst.shape == (batch_size, MC)
    lst = lst.mean(-1)
    assert lst.shape == (batch_size,)
    return lst


def dsm_helper(u_t, t_arr, noise, LT_inv, sde):
    n_vars = sde.config.n_vars
    GGT = sde.GGT(t_arr)
    L_inv = batch_transpose(LT_inv)
    matrix = torch.bmm(L_inv, torch.bmm(GGT, LT_inv))

    data_dim = u_t.shape[1] / n_vars
    quad_forward = trace_fn(matrix) * data_dim

    noise_diff = sde.eps_pred(u_t, t_arr) - noise
    score_diff = block_mat_multiply(-LT_inv, noise_diff, n_vars)
    dsm = block_quadratic_form(score_diff, GGT, score_diff, n_vars)

    div_f = sde.div_f(u_t, t_arr)

    main_term = -0.5 * dsm + 0.5 * quad_forward + div_f

    return main_term


def ism_helper(u_t, t_arr, sde, hutch_mc):
    n_vars = sde.config.n_vars

    # compute div g2s - f
    div_GGs = compute_div_GGs(u_t, t_arr, sde, MC=hutch_mc, method="grad")
    div_term = div_GGs - sde.div_f(u_t, t_arr)

    s = sde.score_fn(u_t, t_arr)
    GGT = sde.GGT(t_arr)
    s_GGT_s = block_quadratic_form(s, GGT, s, n_vars)

    main_term = -0.5 * s_GGT_s - div_term
    return main_term


def ism_or_dsm(u_0, t, sde, elbo_type, hutch_mc=None):
    hybrid = sde.config.hybrid_transition_kernel

    u_t, noise, LT_inv = sde.sample_from_transition_kernel(u_0, t, hybrid=hybrid)

    if elbo_type == "ism_elbo":
        main_term = ism_helper(u_t, t, sde, hutch_mc)
    elif elbo_type == "dsm_elbo":
        main_term = dsm_helper(u_t, t, noise, LT_inv, sde)
    else:
        raise NotImplementedError(f"Loss type {elbo_type} not implemented")

    return main_term


def cross_entropy(u_0, sde):
    hybrid = sde.config.hybrid_transition_kernel

    batch_size = u_0.shape[0]
    T = torch.ones(batch_size).to(u_0) * sde.config.T_max
    u_T, *_ = sde.sample_from_transition_kernel(u_0, t_arr=T, hybrid=hybrid)
    log_pi_uT = sde.prior_logp(u_T)
    return log_pi_uT


def logq_v0(sde, v_0):
    n_vars = sde.config.n_vars
    if n_vars == 1:
        assert len(v_0) == 0
        return 0
    else:
        return sde.logq_v0(v_0)


def nelbo(
    batch,
    sde,
    elbo_type,
    importance_weight,
    MC,
    mode,
    return_zeros=False,
    hutch_mc=None,
    fixed_time_array=None,
):
    if return_zeros:
        zero_arr = torch.ones(batch.shape[0]).to(batch) * -10000.0
        return dict(nelbo_offset=zero_arr, nelbo_no_offset=zero_arr)

    batch_size = batch.shape[0]
    elbo_dict = dict(elbo_offset=[], elbo_no_offset=[])

    for _ in range(MC):
        elbo_i = elbo_fn(
            batch=batch,
            sde=sde,
            elbo_type=elbo_type,
            importance_weight=importance_weight,
            mode=mode,
            hutch_mc=hutch_mc,
            fixed_time_array=fixed_time_array,
        )

        for loss_type in elbo_i:
            elbo_dict[loss_type].append(elbo_i[loss_type])

    for loss_type in elbo_dict:
        elbo_dict[loss_type] = torch.stack(elbo_dict[loss_type], dim=1)

        assert elbo_dict[loss_type].shape == (batch_size, MC)
        elbo_dict[loss_type] = elbo_dict[loss_type].mean(1)

        assert elbo_dict[loss_type].shape == (batch_size,)

    nelbo_dict = {"n" + key: -elbo_dict[key] for key in elbo_dict}

    return nelbo_dict


def elbo_fn(
    batch, sde, elbo_type, importance_weight, mode, hutch_mc=None, fixed_time_array=None
):
    assert mode in ["train", "eval"]
    if elbo_type == "ism_elbo":
        assert hutch_mc is not None

    u_0, v_0 = sde.make_u0(batch)
    batch_size, _ = u_0.shape

    if mode == "train":
        t_min, t_max = sde.config.T_min_train, sde.config.T_max
    elif mode == "eval":
        t_min, t_max = sde.config.T_min_eval, sde.config.T_max
    else:
        raise NotImplementedError("mode not implemented")

    if importance_weight:
        sampler = VPImpSamp(
            sde=sde,
            beta_0=sde.config.beta_0,
            beta_1=sde.config.beta_1,
            t_min=t_min,
            t_max=t_max,
        )
        if fixed_time_array is None:
            t = sampler.sample(batch_size).type_as(u_0)
        else:
            t = fixed_time_array
        w = sampler.weight(t)

    else:
        if fixed_time_array is None:
            t_uniform = torch.rand(
                batch_size,
            )
            t = t_uniform * (t_max - t_min) + t_min
            t = t.type_as(u_0)
        else:
            t = fixed_time_array
        w = 1.0

    # ptilde sde
    log_pi_uT = cross_entropy(u_0, sde)
    integral_term = ism_or_dsm(u_0, t, sde, elbo_type, hutch_mc) * (1 - t_min) / w
    elbo_u_eps = log_pi_uT + integral_term

    elbo_no_offset = elbo_u_eps - logq_v0(sde, v_0)
    elbo_offset = elbo_no_offset + offset_term(u_0=u_0, v_0=v_0, sde=sde, eps=t_min)
    return dict(elbo_offset=elbo_offset, elbo_no_offset=elbo_no_offset)


def gaussian_log_prob(var, loc, scale=None, cov=None, n_vars=None):
    data_dim = var.shape[1] // n_vars
    batch_size = var.shape[0]

    # N, n_vars, dim -> N, dim, n_vars
    unpacked_loc = batch_transpose(
        unpack_vars(
            loc,
            n_vars=n_vars,
        )
    )

    assert unpacked_loc.shape == (batch_size, data_dim, n_vars)
    assert cov[0].shape == (n_vars, n_vars)

    q_dist = torch.distributions.MultivariateNormal(
        loc=unpacked_loc, covariance_matrix=cov[0]
    )

    # N, n_vars, dim -> N, dim, n_vars
    unpacked_var = batch_transpose(
        unpack_vars(
            var,
            n_vars=n_vars,
        )
    )
    assert unpacked_var.shape == (batch_size, data_dim, n_vars)

    log_prob = q_dist.log_prob(unpacked_var)

    assert log_prob.shape == (batch_size, data_dim)
    log_prob = log_prob.sum(1)

    assert log_prob.shape == (batch_size,)
    return log_prob


def offset_term(u_0: torch.Tensor, v_0: torch.Tensor, sde, eps: torch.Tensor):
    # in case u_0's v sub-component was modified, put the original v_0 back in
    original_u0_shape = u_0.shape

    if sde.config.n_vars > 1:
        u_0[:, sde.data_dim :] = v_0
        new_u0_shape = u_0.shape
        assert original_u0_shape == new_u0_shape

    config = sde.config
    batch_size, dimu = u_0.shape
    eps_vec = torch.ones(batch_size).to(u_0) * eps

    # log variational
    # q(u_eps|u_0) = N(u_eps ; mean_coef_eps u_0, var_eps I)
    u_eps, _, _ = sde.sample_from_transition_kernel(u_0, eps_vec, hybrid=False)
    mu_eps, _ = sde.transition_mean(u_0, eps_vec, hybrid=False)

    # this thing might be tiny for alda
    var_eps = sde.transition_var(eps_vec, hybrid=False)

    assert u_eps.shape == (batch_size, dimu)
    assert mu_eps.shape == (batch_size, dimu)

    if config.sde_type == "vpsde":
        var_eps = var_eps.unsqueeze(2)

    assert var_eps.shape == (batch_size, config.n_vars, config.n_vars)
    assert u_eps.shape == (batch_size, dimu)
    assert mu_eps.shape == (batch_size, dimu)

    lq = gaussian_log_prob(var=u_eps, loc=mu_eps, cov=var_eps, n_vars=config.n_vars)

    ll_mean, ll_cov = sde.offset_likelihood(u_eps, eps_vec)

    # NOTE the covariance matrix is assumed to be the same across batches and data dimensions
    ll = gaussian_log_prob(var=u_0, loc=ll_mean, cov=ll_cov, n_vars=config.n_vars)

    return ll - lq


def handle_noise_pred(data, sde, loss_type):
    batch_size = data.shape[0]

    hybrid = sde.config.hybrid_transition_kernel

    t_min, t_max = sde.config.T_min_train, sde.config.T_max
    t_uniform = torch.rand(
        batch_size,
    )
    t_arr = t_uniform * (t_max - t_min) + t_min
    t_arr = t_arr.type_as(data)

    u_0, _ = sde.make_u0(data)
    u_t, eps, _ = sde.sample_from_transition_kernel(u_0=u_0, t_arr=t_arr, hybrid=hybrid)

    eps_pred = sde.eps_pred(u_t, t_arr)

    if loss_type == "cld_noise_pred":
        # in principle, works for any aux var system with no noise in data.
        assert sde.config.sde_type in ["cld", "malda", "alda"]
        eps_idx = 2 * sde.data_dim if sde.config.sde_type == "alda" else sde.data_dim

        eps = eps[:, eps_idx:]
        eps_pred = eps_pred[:, eps_idx:]

    loss = torch.pow(eps - eps_pred, exponent=2).sum(1)
    return loss


def train_loss_fn(data, sde, loss_type):
    if loss_type in ["noise_pred", "cld_noise_pred"]:
        loss = handle_noise_pred(data, sde, loss_type)

    elif loss_type in ["ism_elbo", "dsm_elbo"]:
        nelbo_dict = nelbo(
            batch=data,
            sde=sde,
            elbo_type=loss_type,
            importance_weight=sde.config.imp_weight_train,
            MC=sde.config.elbo_mc_samples_train,
            mode="train",
            hutch_mc=None,
            fixed_time_array=None,
        )
        loss = nelbo_dict["nelbo_no_offset"]

    else:
        raise NotImplementedError(f"Loss type {loss_type} not implemented")

    return loss
