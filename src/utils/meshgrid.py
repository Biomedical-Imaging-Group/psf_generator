import typing

import torch

from params import Params


def meshgrid_pupil(params: Params) -> typing.Tuple[torch.Tensor, torch.Tensor]:
    """ function to define the meshgrid for the 2D pupil plane
    The radius of the aperture/pupil plane defines the actual physical size of the image.
    The parameter -size- specifies the number of grid points sampled on the pupil plane.

    Parameters
    ----------
    params: Params
        class of parameters used in this library
    Returns
    -------
    xx, yy
        coordinates in the Fourier domain (kx, ky)
    """
    size = params.get('n_pix_pupil')
    radius = params.get('pupil_fov_phy') / 2
    device = params.device
    x = torch.linspace(-radius, radius, size).to(device)
    y = torch.linspace(-radius, radius, size).to(device)
    xx, yy = torch.meshgrid(x, y, indexing='xy')
    return xx, yy
