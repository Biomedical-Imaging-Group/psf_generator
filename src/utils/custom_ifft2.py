import torch

from params import Params
from utils.czt import Czt1d


def custom_ifft2(pupil: torch.Tensor, params: Params):
    """ifft2 but with zooming
    This function generalizes torch.ifft2 in the sense that it allows
    1. the size of the input and output images are different
    2. zooming into a certain region in the output image
    The above nice features are enabled by the chirp Z transform defined in utils.czt.Czt1d().
    In particular, the zooming needs to be taken properly into account in the czt by multiplying a correction factor
    which is similar to the effect of (i)fftshift.
    Parameters
    ----------
    pupil: torch.Tensor
        input 2D image to be transformed
    params: Params
        parameters

    Returns
    -------
    psf: torch.Tensor
        transformed images of desired output size and zooming
    """
    N = params.get('n_pix_pupil')
    M = params.get('n_pix_psf')
    w = params.get('czt_w')
    a = params.get('czt_a')
    k = torch.arange(M)
    zoom = params.get('psf_zooming_factor')
    device = params.device
    correction = torch.exp(-1j * torch.pi * zoom * (k - M / 2)).to(device)

    czt1d = Czt1d(N, M, w, a, mode='idft', device=device)
    output = torch.vstack([czt1d(pupil[row, :]) * correction for row in range(N)])
    psf = torch.column_stack([czt1d(output[:, col]) * correction for col in range(M)])

    return psf
