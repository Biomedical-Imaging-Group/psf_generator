# Copyright Biomedical Imaging Group, EPFL 2025

"""
The propagator for the vectorial field in the spherical coordinates.
"""

import math
import typing as tp

import torch
from torch import vmap
from torch.special import bessel_j0, bessel_j1

from .spherical_propagator import SphericalPropagator
from ..utils.integrate import simpsons_rule


class VectorialSphericalPropagator(SphericalPropagator):
    r"""
    Propagator for the vectorial case of the Richard's-Wolf integral in spherical parameterization.

    The equation to compute the electric field is

    .. math::
            \mathbf{E}(\boldsymbol{\rho}) =
            - \frac{\mathrm{i} fk}{2}
            \begin{bmatrix}
                {2}^y\sin2\varphi\\
                - I_{2}^x\sin2\varphi + [I_{0}^y + I_{2}^y\cos2\varphi]\\
                -2\mathrm{i} I_{1}^x\cos\varphi  - 2\mathrm{i} I_{1}^y\sin\varphi
            \end{bmatrix},

    where

    .. math::
            I_{0}^a (\rho,z) =
            \int_0^{\theta_{\max}} \boldsymbol{e}_{\textrm{inc}^a}(\theta)\sin\theta (\cos\theta+1)
            J_0(k\rho\sin\theta)\mathrm{e}^{\mathrm{i} kz\cos\theta}d\theta,

            I_{1}^a (\rho,z)=
            \int_0^{\theta_{\max}} \boldsymbol{e}_{\textrm{inc}^a}(\theta)\sin^2\theta
            J_1(k\rho\sin\theta)\mathrm{e}^{\mathrm{i} kz\cos\theta}d\theta,

            I_{2}^a (\rho,z) =
            \int_0^{\theta_{\max}} \boldsymbol{e}_{\textrm{inc}^a}(\theta)\sin\theta (\cos\theta-1)
            J_2(k\rho\sin\theta)\mathrm{e}^{\mathrm{i} kz\cos\theta}d\theta,

    where :math:`a\in\{x,y\}, \boldsymbol{e}_{\textrm{inc}}(\theta) =
    [\boldsymbol{e}_{\textrm{inc}}^x(\theta), \boldsymbol{e}_{\textrm{inc}}^y(\theta), 0]`.

    Parameters
    ----------
    `self.e0x` : float, optional
        Initial electric field component :math:`\mathbf{e}_0^x`. Default value is `1.0`.
    `self.e0y` : float, optional
        Initial electric field component :math:`\mathbf{e}_0^y`. Default value is `0.0`.

    Notes
    -----
    The vectorial propagators have two additional arguments apart from those inherited form the base propagator
    to account for polarization.

    """

    def __init__(self, n_pix_pupil=128, n_pix_psf=128, device='cpu',
                 zernike_coefficients=None,
                 e0x=1.0, e0y=0.0,
                 wavelength=632, na=1.3, fov=1000,
                 defocus_min=0, defocus_max=0, n_defocus=1,
                 apod_factor=False, envelope=None, cos_factor=False,
                 gibson_lanni=False, z_p=1e3, n_s=1.3,
                 n_g=1.5, n_g0=1.5, t_g=170e3, t_g0=170e3,
                 n_i=1.5, n_i0=1.5, t_i0=100e3,
                 integrator=simpsons_rule):
        super().__init__(n_pix_pupil=n_pix_pupil, n_pix_psf=n_pix_psf, device=device,
                         zernike_coefficients=zernike_coefficients,
                         wavelength=wavelength, na=na, fov=fov,
                         defocus_min=defocus_min, defocus_max=defocus_max, n_defocus=n_defocus,
                         apod_factor=apod_factor, envelope=envelope, cos_factor=cos_factor,
                         gibson_lanni=gibson_lanni, z_p=z_p, n_s=n_s,
                         n_g=n_g, n_g0=n_g0, t_g=t_g, t_g0=t_g0,
                         n_i=n_i, n_i0=n_i0, t_i0=t_i0,
                         integrator=integrator)

        self.e0x = e0x
        self.e0y = e0y
        # PSF varphi coordinate
        varphi = torch.atan2(self.yy, self.xx)
        sin_phi = torch.sin(varphi)
        cos_phi = torch.cos(varphi)
        sin_twophi = 2.0 * sin_phi * cos_phi
        cos_twophi = cos_phi ** 2 - sin_phi ** 2
        self.sin_phi = sin_phi.to(self.device)
        self.cos_phi = cos_phi.to(self.device)
        self.sin_twophi = sin_twophi.to(self.device)
        self.cos_twophi = cos_twophi.to(self.device)

    @classmethod
    def get_name(cls) -> str:
        return 'vectorial_spherical'

    def _get_args(self) -> tp.Dict:
        args = super()._get_args()
        args['e0x'] = str(self.e0x)
        args['e0y'] = str(self.e0y)
        return args

    def get_input_field(self) -> torch.Tensor:
        single_field = torch.ones(self.n_pix_pupil).to(self.device)
        input_field = torch.stack((self.e0x * single_field, self.e0y * single_field),
                           dim=0).to(torch.complex64)
        return input_field

    def compute_focus_field(self) -> torch.Tensor:
        """
        Compute the focus field.

        Returns
        -------
        field: torch.Tensor
            Output PSF.

        Notes
        -----
        This involves expensive evaluations of Bessel functions.
        We compute it independently of defocus and handle defocus via batching with `vmap()`.

        """
        sin_t = torch.sin(self.thetas)
        cos_t = torch.cos(self.thetas)
        bessel_arg = self.k * self.rs[None, :] * sin_t[:, None] * self.refractive_index
        J0 = bessel_j0(bessel_arg)
        J1 = bessel_j1(bessel_arg)
        J2 = 2.0 * torch.where(bessel_arg > 1e-6, J1 / bessel_arg, 0.5 - bessel_arg ** 2 / 16) - J0

        batched_compute_field_at_defocus = vmap(self._compute_psf_at_defocus,
                                                in_dims=(0, None, None, None, None, None, None))
        return batched_compute_field_at_defocus(self.defocus_filters, J0, J1, J2, self.get_pupil(), sin_t, cos_t)


    def _compute_psf_at_defocus(
            self,
            defocus_term: torch.Tensor,
            J0: torch.Tensor,
            J1: torch.Tensor,
            J2: torch.Tensor,
            pupil: torch.Tensor,
            sin_t: torch.Tensor,
            cos_t: torch.Tensor,
    ) -> torch.Tensor:
        r"""
        Compute the PSF at defocus.

        Parameters
        ----------
        defocus_term: torch.Tensor
            Factor in the integrand corresponding to defocus.
        J0: torch.Tensor
            Bessel function of the first kind of order 0 :math:`J_0`.
        J1: torch.Tensor
            Bessel function of the first kind of order 1 :math:`J_1`.
        J2: torch.Tensor
            Bessel function of the first kind of order 2 :math:`J_2`.
        pupil: torch.Tensor
            Pupil function.
        sin_t: torch.Tensor
            shape: (n_thetas, )
        cos_t: torch.Tensor
            shape: (n_thetas, )

        Returns
        -------
        PSF_field: torch.Tensor
            Output field.

        """
        field_x, field_y = pupil[0, :], pupil[1, :]

        Is = []
        fixed_factor = sin_t * defocus_term
        factors = [(cos_t + 1.0), sin_t, (cos_t - 1.0)]
        for bessel, factor in zip([J0, J1, J2], factors):
            for field in [field_x, field_y]:
                I_term = fixed_factor * factor
                item = self.integrator(fs=bessel * (field * I_term)[:, None], dx=self.dtheta)
                item = item[self.rr_indices]
                Is.append(item)
        Ix0, Iy0, Ix1, Iy1, Ix2, Iy2 = Is

        PSF_field = torch.stack([
            Ix0 - Ix2 * self.cos_twophi - Iy2 * self.sin_twophi,
            Iy0 - Ix2 * self.sin_twophi + Iy2 * self.cos_twophi,
            -2j * (Ix1 * self.cos_phi + Iy1 * self.sin_phi)],
            dim=0)

        return PSF_field / 2 / math.sqrt(self.refractive_index)
