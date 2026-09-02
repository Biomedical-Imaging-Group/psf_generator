# Copyright Biomedical Imaging Group, EPFL 2025

"""
The dipole imager in Cartesian coordinates (chirp Z transform).
"""
import math
import warnings

import torch

from .dipole_imager import DipoleImager
from ..utils.czt import custom_ifft2
from ..utils.zernike import create_special_pupil


class CartesianDipoleImager(DipoleImager):
    r"""
    Dipole imager in Cartesian coordinates: the pupil field is built on a square grid and the image is its
    two-dimensional Fourier transform, evaluated on the image grid with a chirp Z transform.

    Unlike the spherical imager, this imager supports arbitrary (non-axisymmetric) pupil functions: any Zernike
    mode, a special phase mask or a custom pupil factor of the detection path.

    Parameters
    ----------
    special_phase_mask : str or torch.Tensor, optional
        Special phase mask of the pupil, see :func:`psf_generator.utils.zernike.create_special_pupil`.
    custom_field : torch.Tensor or array-like, optional
        Extra complex factor applied to the pupil, of shape `(n_pix_pupil, n_pix_pupil)`.

    See :class:`DipoleImager` for the other parameters.

    Notes
    -----
    The pupil grid is ``torch.linspace(-1, 1, n_pix_pupil)`` in the normalized coordinate
    :math:`s / s_{\max}` along both axes (dim 0 is :math:`s_y`, dim 1 is :math:`s_x`), like for the Cartesian
    propagators. The image of a dipole at :math:`(x_p, y_p)` is obtained with the phase ramp
    :math:`\mathrm{e}^{-\mathrm{i} k_i (s_x x_p + s_y y_p)}` on the pupil.

    """

    _zernike_mesh_type = 'cartesian'

    def __init__(self, n_pix_pupil=128, n_pix_psf=128, device='cpu',
                 zernike_coefficients=None,
                 special_phase_mask=None,
                 custom_field=None,
                 wavelength=632, na=1.3, pix_size=20, z_focus=0.0,
                 n_s=1.33, n_g=1.5, n_g0=1.5, t_g=170e3, t_g0=170e3,
                 n_i=1.5, n_i0=1.5, t_i0=100e3,
                 fresnel='reciprocal'):
        super().__init__(n_pix_pupil=n_pix_pupil, n_pix_psf=n_pix_psf, device=device,
                         zernike_coefficients=zernike_coefficients,
                         wavelength=wavelength, na=na, pix_size=pix_size, z_focus=z_focus,
                         n_s=n_s, n_g=n_g, n_g0=n_g0, t_g=t_g, t_g0=t_g0,
                         n_i=n_i, n_i0=n_i0, t_i0=t_i0, fresnel=fresnel)
        n = n_pix_pupil
        # Pupil grid in the normalized coordinate s / s_max (dim 0 is s_y, dim 1 is s_x).
        s = torch.linspace(-1.0, 1.0, n, dtype=torch.float64)
        self.ds = 2.0 / (n - 1)
        s_yy, s_xx = torch.meshgrid(s, s, indexing='ij')
        sin_t = self.s_max * torch.sqrt(s_xx ** 2 + s_yy ** 2)
        mask = sin_t <= self.s_max
        phi = torch.atan2(s_yy, s_xx)
        cos_phi, sin_phi = torch.cos(phi), torch.sin(phi)
        cos_2phi, sin_2phi = cos_phi ** 2 - sin_phi ** 2, 2.0 * sin_phi * cos_phi

        factors = self._layer_factors(sin_t.clamp(max=1.0))
        # Apodization 1/sqrt(cos) (sphere to plane), no 1/s_z Jacobian; the cosine is clamped at the rim like
        # in the Cartesian propagators (na == n_i0 gives cos = 0 on the edge of the pupil).
        apodization = 1.0 / torch.sqrt(factors['cos_t'].clamp(min=1e-3))
        base = torch.where(mask, torch.exp(1j * self.k * factors['path0']) * apodization,
                           torch.zeros((), dtype=torch.complex128))
        a_0, a_1, a_2 = self._pattern_factors(factors)
        # Coefficients of (p_x, p_y, p_z) in the x and y components of the pupil field.
        pattern_x = torch.stack([0.5 * (a_0 + a_2 * cos_2phi), 0.5 * a_2 * sin_2phi, a_1 * cos_phi])
        pattern_y = torch.stack([0.5 * a_2 * sin_2phi, 0.5 * (a_0 - a_2 * cos_2phi), a_1 * sin_phi])
        pattern = torch.stack([pattern_x, pattern_y]) * base                          # [2, 3, n, n]
        pattern = torch.where(mask[None, None], pattern, torch.zeros((), dtype=torch.complex128))
        self._pattern = pattern.to(torch.complex64).to(self.device)
        self._path_z = torch.where(mask, factors['path_z'], torch.zeros((), dtype=torch.complex128))
        self._path_z = self._path_z.to(torch.complex64).to(self.device)
        # Phase per nanometer of lateral displacement of the dipole.
        self._ramp_x = (-self.k * self.n_i * self.s_max * s_xx).to(torch.float32).to(self.device)
        self._ramp_y = (-self.k * self.n_i * self.s_max * s_yy).to(torch.float32).to(self.device)

        # Chirp Z transform mapping the pupil onto the image grid, see ``CartesianPropagator``.
        phase_per_pupil_step = self.k * self.n_i * self.s_max * self.ds
        self.k_start = phase_per_pupil_step * float(self.x[0])
        self.k_end = phase_per_pupil_step * float(self.x[-1])

        self.special_phase_mask = special_phase_mask
        self._special_pupil = create_special_pupil(n, mask=special_phase_mask).to(self.device)
        self.update_custom_field(custom_field)
        self._compute_zernike_aberrations()

    @classmethod
    def get_name(cls) -> str:
        return 'cartesian'

    def update_custom_field(self, custom_field) -> None:
        """
        Update the custom pupil factor without reinitializing the imager.

        Parameters
        ----------
        custom_field : torch.Tensor or None
            Complex factor of shape (n_pix_pupil, n_pix_pupil) or (1, 1, n_pix_pupil, n_pix_pupil), or None.
        """
        if custom_field is None:
            self.custom_field = None
            return
        if not isinstance(custom_field, torch.Tensor):
            custom_field = torch.tensor(custom_field, dtype=torch.complex64)
        n = self.n_pix_pupil
        if custom_field.shape == (1, 1, n, n):
            custom_field = custom_field.reshape(n, n)
        if custom_field.shape != (n, n):
            raise ValueError(f'custom_field must have shape ({n}, {n}) or (1, 1, {n}, {n})')
        self.custom_field = custom_field.to(torch.complex64).to(self.device)

    def get_pupil_factor(self) -> torch.Tensor:
        """Zernike, special-mask and custom factors of the pupil, of shape `(n_pix_pupil, n_pix_pupil)`."""
        factor = self._zernike_aberrations * self._special_pupil
        if self.custom_field is not None:
            factor = factor * self.custom_field
        return factor

    def _get_args(self) -> dict:
        args = super()._get_args()
        special_phase_mask = self.special_phase_mask
        if isinstance(special_phase_mask, torch.Tensor):
            warnings.warn('A custom special_phase_mask tensor cannot be saved to JSON; it is stored as None.',
                          stacklevel=3)
            special_phase_mask = None
        args['special_phase_mask'] = special_phase_mask
        self._warn_custom_field_not_saved()
        return args

    def _pupil_xy(self, dipole: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Transverse pupil field of every position, of shape `(n_positions, 2, n_pix_pupil, n_pix_pupil)`."""
        device = self.device
        dipole = dipole.to(torch.complex64).to(device)
        # [2, n, n]: pattern of the dipole in the pupil plane, then the common pupil factors.
        pupil = torch.einsum('c,dcij->dij', dipole, self._pattern) * self.get_pupil_factor()[None]
        positions = positions.to(torch.float32).to(device)
        x_p, y_p = positions[:, 0].reshape(-1, 1, 1), positions[:, 1].reshape(-1, 1, 1)
        z_p = positions[:, 2].reshape(-1, 1, 1).to(torch.complex64)
        # Lateral shift (phase ramp) and height above the coverslip (optical path).
        shift = torch.exp(1j * (x_p * self._ramp_x[None] + y_p * self._ramp_y[None]))
        axial = torch.exp(1j * self.k * z_p * self._path_z[None])
        return pupil[None] * (shift * axial)[:, None]

    def get_pupil(self, dipole=(1.0, 0.0, 0.0), positions=(0.0, 0.0, 0.0)) -> torch.Tensor:
        """
        Get the pupil field (collimated beam after the objective) with all factors applied.

        Parameters
        ----------
        dipole, positions
            See :meth:`compute_image`.

        Returns
        -------
        pupil : torch.Tensor
            Complex field of shape `(n_positions, 3, n_pix_pupil, n_pix_pupil)`; the third component is zero.

        """
        pupil_xy = self._pupil_xy(self._as_dipole(dipole), self._as_positions(positions))
        return torch.cat([pupil_xy, torch.zeros_like(pupil_xy[:, :1])], dim=1)

    def _compute_image_xy(self, dipole: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        pupil = self._pupil_xy(dipole, positions)
        field = custom_ifft2(pupil, shape_out=(self.n_pix_psf, self.n_pix_psf),
                             k_start=self.k_start, k_end=self.k_end,
                             norm='forward', fftshift_input=True, include_end=True) * (self.ds * self.s_max) ** 2
        return field * (-1j * self.k * self.n_i / (2.0 * math.pi))
