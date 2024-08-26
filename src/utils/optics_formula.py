import torch


def optical_path(z_p, n_s,
                 n_g, n_g0,
                 t_g, t_g0,
                 n_i, n_i0,
                 t_i, t_i0,
                 sin_t):
    # computed following Eq. (3.45) of François Aguet's thesis
    path = z_p * torch.sqrt(n_s ** 2 - n_i ** 2 * sin_t ** 2) \
                   + t_i * torch.sqrt(n_i ** 2 - n_i ** 2 * sin_t ** 2) \
                   - t_i0 * torch.sqrt(n_i0 ** 2 - n_i ** 2 * sin_t ** 2) \
                   + t_g * torch.sqrt(n_g ** 2 - n_i ** 2 * sin_t ** 2) \
                   - t_g0 * torch.sqrt(n_g0 ** 2 - n_i ** 2 * sin_t ** 2)
    return path