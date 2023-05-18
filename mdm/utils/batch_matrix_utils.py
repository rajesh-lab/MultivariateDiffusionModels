import torch


def trace_fn(A):
    return A.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1)


def safe_cholesky(A, sde):
    try:
        L = torch.linalg.cholesky(A)
    except:
        assert len(A.shape) == 3
        bsz = A.shape[0]
        n_vars = A.shape[1]
        I = torch.eye(n_vars).to(A)
        I = I.unsqueeze(0).repeat(bsz, 1, 1)
        assert A.shape == I.shape
        eps = 0.0001

        try:
            L = torch.linalg.cholesky(A + eps * I)
        except torch._C._LinAlgError:
            A_eps_I = A + eps * I
            for i in range(A_eps_I.shape[0]):
                mat = A_eps_I[i]
                try:
                    Li = torch.linalg.cholesky(mat)
                except:
                    print("failed")
                    print("cov[i] was ", mat)
                    print("Q was", sde.get_Q())
                    print("D was", sde.get_D())
                    assert False
            assert False

    return L


def pack_vars(arr_vars, n_vars) -> torch.Tensor:
    """
    input: [x, v_1, ...., v_n] (n_batches, n_vars, data_dim)
    return:  u (n_batches, dim * n_vars)
    """
    arr_vars = torch.chunk(arr_vars, chunks=n_vars, dim=1)
    u = torch.cat(arr_vars, dim=-1).squeeze(1)

    # assert u.shape == (arr_vars.shape[0], n_vars, arr_vars.shape[1] // n_vars)

    return u


def matrix_exp(t, A):
    t = t.unsqueeze(-1).unsqueeze(-1)
    mexp = torch.matrix_exp(t * A)
    return mexp


def unpack_vars(u: torch.Tensor, n_vars: int) -> torch.Tensor:
    """
    input: u (n_batches, dim * n_vars)
    return: [x, v_1, ...., v_n] (n_batches, n_vars, dim)
    """
    arr_vars = torch.chunk(u, chunks=n_vars, dim=1)
    arr_vars = torch.stack(arr_vars, dim=1)

    return arr_vars


def block_mat_multiply(A: torch.Tensor, u: torch.Tensor, n_vars: int) -> torch.Tensor:
    """
    input: A (n_batches, n_vars, n_vars)
    input: u (n_batches, n_vars * input_shape)

    returns: A y (n_batches, n_vars * input_shape)
    """
    arr_vars = unpack_vars(u, n_vars=n_vars)
    Au = torch.bmm(A, arr_vars)

    return pack_vars(Au, n_vars=n_vars)


def block_mat_mat_multiply(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    input: A (n_batches, n_vars, n_vars)
    input: B (n_batches, n_vars, n_vars)

    returns: A @ B (n_batches, n_vars, n_vars)
    """
    AB = torch.bmm(A, B)

    return AB


def block_quadratic_form(
    x: torch.Tensor,
    A: torch.Tensor,
    y: torch.Tensor,
    n_vars: int,
) -> torch.Tensor:
    """
    input: x (n_batches, n_vars * input_shape)
    input: A (n_batches, n_vars, n_vars)
    input: y (n_batches, n_vars * input_shape)

    returns: x^T A y (n_batches,)
    """
    assert x.shape == y.shape
    n_batches, _ = y.shape

    arr_vars_y = unpack_vars(y, n_vars=n_vars)

    Ay = torch.bmm(A, arr_vars_y)
    Ay = pack_vars(Ay, n_vars=n_vars)

    x_t_Ay = torch.sum(x * Ay.squeeze(1), dim=1)

    assert x_t_Ay.shape == (n_batches,)

    return x_t_Ay


def batch_dot_product(x, y):
    x = x.unsqueeze(1)
    y = y.unsqueeze(2)

    xTy = torch.bmm(x, y)

    return xTy.squeeze(-1).squeeze(-1)


def batched_scalar_mat(A: torch.Tensor, t: torch.Tensor):
    t = t.unsqueeze(-1).unsqueeze(1)
    return A * t
