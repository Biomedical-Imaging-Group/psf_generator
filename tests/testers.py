"""
This file contains several classes for measuring the accuracy and convergence of the scalar propagators.
"""
from __future__ import annotations

from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.integrate import quad, quad_vec
from scipy.special import j0 as sp_j0
from scipy.special import j1 as sp_j1
from torch.special import bessel_j0, bessel_j1
from tqdm import tqdm

from propagators import (
    ScalarCartesianPropagator,
    ScalarPolarPropagator,
    VectorialCartesianPropagator,
    VectorialPolarPropagator,
)
from pupil import (
    ScalarCartesianPupil,
    ScalarPolarPupil,
    VectorialCartesianPupil,
    VectorialPolarPupil,
)

from utils.zernike import index_to_nl, zernike_nl


def sp_j2(x: np.ndarray) -> np.ndarray:
    return 2.0 * np.where(np.abs(x) > 1e-6, sp_j1(x) / x, 0.5 - x ** 2 / 16) - sp_j0(x)


class TestCase:
    def __init__(self,
                 name: str=None,
                 psf_expr: Callable=None,
                 ):
        self.name = name

        self.use_analytic_psf = (psf_expr is not None)
        self.psf_expr = psf_expr

    def get_name(self):
        return self.name

    def eval_pupil_at(self, sin_t: torch.Tensor, sin_t_max: float) -> torch.Tensor:
        """
        Evaluate the pupil field $e_{\infty}(\sin{\theta})$ at the pupil coordinate $\sin{\theta}$.
        """
        raise NotImplementedError

    def eval_pupil_at_np(self, sin_t: np.ndarray, sin_t_max: float) -> np.ndarray:
        """
        Evaluate the pupil field $e_{\infty}(\sin{\theta})$ at the pupil coordinate $\sin{\theta}$.
        This version is implemented in numpy and is used to generate the ground truth PSF field.
        """
        raise NotImplementedError

    def eval_PSF_at(self, kr: torch.Tensor, sin_t_max: float) -> torch.Tensor:
        """
        Evaluate the PSF at the normalized PSF radial coordinate $k * r$, which is the product of the
        radial position `r` and the wavenumber `k`.
        """
        psf_coord = kr
        if self.use_analytic_psf:
            psf_field = self.psf_expr(psf_coord)
        else:
            psf_field = _J_integral_scalar(
                lambda sin_t: self.eval_pupil_at_np(sin_t, sin_t_max), 
                sin_t_max,
                psf_coord)
        return psf_field

    def get_fields_as_polar(self, thetas: torch.Tensor, krs: torch.Tensor, sin_t_max: float):
        """
        Evaluate the test case on a polar grid.

        Inputs:
        - thetas: torch.Tensor[n_pix_pupil,]. The pupil coordinates where the pupil far-field
                is evaluated.
        - krs: torch.Tensor[n_pix_psf,]. The PSF radial coordinates where the PSF is evaluated.
                This is equal to the product of the radial position `r` and the wavenumber `k`.

        Outputs:
        - pupil_field: torch.Tensor[n_pix_pupil,]. The evaluated pupil field. This is
                        used as input for the propagator's numerical integration routine.
        - psf_field: torch.Tensor[n_pix_psf,]. The evaluated PSF field.
        """
        sin_t = torch.sin(thetas)
        pupil_field = self.eval_pupil_at(sin_t, sin_t_max)
        psf_field = self.eval_PSF_at(krs, sin_t_max)
        return pupil_field, psf_field

    def get_fields_as_cartesian(self,
                      s_x: torch.Tensor,
                      s_y: torch.Tensor,
                      norm_x: torch.Tensor,  # == k * x
                      norm_y: torch.Tensor,  # == k * y
                      sin_t_max: float,
                      ):
        """
        Evaluate the test case on a 2D Cartesian grid.

        Inputs:
        - s_x, s_y: torch.Tensor[n_pix_pupil,]. The pupil coordinates where the pupil far-field
                    is evaluated.
        - s_max: float. The maximum input direction for the pupil field (== sin(theta_max)).
        - x, y: torch.Tensor[n_pix_pupil,]. The PSF Cartesian coordinates where the PSF is evaluated,
                normalized by multiplying with the wavenumber `k`:
                    norm_x = k * x
                    norm_y = k * y

        Outputs:
        - pupil_field: torch.Tensor[n_pix_pupil,]. The evaluated pupil field. This is
                        used as input for the propagator's numerical integration routine.
        - psf_field: torch.Tensosr[n_pix_psf,]. The evaluated PSF field.
        """
        s_xx, s_yy = torch.meshgrid(s_x, s_y, indexing='ij')

        sin_t_sq = s_xx ** 2 + s_yy ** 2
        s_valid = sin_t_sq <= sin_t_max ** 2
        sin_ts = torch.where(s_valid, torch.sqrt(sin_t_sq), 0.0)
        pupil_field = self.eval_pupil_at(sin_ts.ravel(), sin_t_max).reshape(sin_ts.shape)
        pupil_field *= s_valid

        xx, yy = torch.meshgrid(norm_x, norm_y, indexing='ij')
        krr = torch.sqrt(xx ** 2 + yy ** 2)

        krs, rr_indices = torch.unique(krr, return_inverse=True)
        psf_field = self.eval_PSF_at(krs, sin_t_max)
        psf_field = psf_field[rr_indices]

        return pupil_field, psf_field


class HankelCase(TestCase):
    """
    Analytic test cases defined using function <-> Hankel transform pairs. For a radial function
    and its Hankel transform pair, {f(u), Hf(v)}, we have the following relation:

        e_inf(\sin{\theta}) = f(\sin{\theta}) * \cos{\theta}    <==>    E_psf(r) = Hf(r)
    """
    def __init__(self,
                 hankel_expr: Callable,
                 hankel_expr_np: Callable=None,
                 psf_expr: Callable=None,
                 Rmax: float = 1.0,
                 name: str=None):
        super().__init__(psf_expr=psf_expr, name=name)

        self.Rmax = Rmax
        self.hankel_expr = hankel_expr
        if hankel_expr_np is not None:
            self.hankel_expr_np = hankel_expr_np
        else:
            self.hankel_expr_np = lambda x: torch.tensor(hankel_expr(x.numpy()))

    def eval_pupil_at(self, sin_t: torch.Tensor, sin_t_max: float) -> torch.Tensor:
        Rmax = min(self.Rmax, sin_t_max)
        cos_t = np.where(sin_t <= sin_t_max, np.sqrt(1.0 - sin_t ** 2), 0.0)
        pupil_coord = sin_t / Rmax
        return self.hankel_expr(pupil_coord) * cos_t / Rmax ** 2

    def eval_pupil_at_np(self, sin_t: np.ndarray, sin_t_max: float) -> np.ndarray:
        Rmax = min(self.Rmax, sin_t_max)
        cos_t = np.where(sin_t <= sin_t_max, np.sqrt(1.0 - sin_t ** 2), 0.0)
        pupil_coord = sin_t / Rmax
        return self.hankel_expr_np(pupil_coord) * cos_t / Rmax ** 2

    def eval_PSF_at(self, kr: torch.Tensor, sin_t_max: float) -> torch.Tensor:
        Rmax = min(self.Rmax, sin_t_max)
        if self.use_analytic_psf:
            psf_field = self.psf_expr(kr * Rmax)
        else:
            psf_field = _J_integral_scalar(
                lambda sin_t: self.eval_pupil_at_np(sin_t, sin_t_max), 
                sin_t_max,
                kr)
        return psf_field


class ScalarPupilCase(TestCase):
    """
    Test cases defined by an input pupil and its Zernike aberrations.
    """
    def __init__(self,
                 zernike_coefficients: np.ndarray,
                 name: str = "pupil_aberrations"):
        self.pupil = ScalarPolarPupil(zernike_coefficients=zernike_coefficients)

        super().__init__(
            name=name,
            psf_expr=None,
        )

    def eval_pupil_at(self, sin_t: torch.Tensor, sin_t_max: float) -> torch.Tensor:
        """Evaluate the pupil field at the radius `rho` = `r`.

        to query the zernike aberrations, normalize the pupil coordinate
        by sin_t_max such that it spans the unit disk [0,1].
        (the Pupil class operates in these normalized coordinates)

        """
        r = sin_t / sin_t_max
        n_zernike = len(self.pupil.zernike_coefficients)
        zernike_phase = torch.zeros_like(r)
        for i in range(n_zernike):
            n, l = index_to_nl(index=i)
            curr_coef = self.pupil.zernike_coefficients[i]
            if l != 0 and curr_coef != 0:
                print("Warning: Zernike coefficients for l != 0 are not supported in polar coordinates.")
            elif l == 0:
                zernike_phase += curr_coef * torch.tensor(zernike_nl(n=n, l=l, rho=r, phi=0.0))
        return torch.exp(1j * zernike_phase).to(self.pupil.device)

    def eval_pupil_at_np(self, sin_t: np.ndarray, sin_t_max: float) -> np.ndarray:
        """Evaluate the pupil field at the radius `rho` = `r`.

        This version is implemented in numpy and is
        used to generate the ground truth PSF field.

        to query the zernike aberrations, normalize the pupil coordinate
        by sin_t_max such that it spans the unit disk [0,1].
        (the Pupil class operates in these normalized coordinates)

        """
        r = sin_t / sin_t_max
        n_zernike = len(self.pupil.zernike_coefficients)
        zernike_phase = np.zeros_like(r)
        for i in range(n_zernike):
            n, l = index_to_nl(index=i)
            curr_coef = self.pupil.zernike_coefficients[i]
            if l != 0 and curr_coef != 0:
                print("Warning: Zernike coefficients for l != 0 are not supported in polar coordinates.")
            elif l == 0:
                zernike_phase += curr_coef * zernike_nl(n=n, l=l, rho=r, phi=0.0)
        return np.exp(1j * zernike_phase)

    def eval_PSF_at(self, kr: torch.Tensor, sin_t_max: float) -> torch.Tensor:
        psf_coord = kr
        psf_field = torch.zeros_like(psf_coord, dtype=torch.complex64)

        psf_field.real = _J_integral_scalar(
                lambda sin_t: self.eval_pupil_at_np(sin_t, sin_t_max).real,
                sin_t_max,
                psf_coord,
                )

        psf_field.imag = _J_integral_scalar(
                lambda sin_t: self.eval_pupil_at_np(sin_t, sin_t_max).imag,
                sin_t_max,
                psf_coord,
                )

        return psf_field


class VectorPupilCase(TestCase):
    """
    Test case for a VectorialPupil and its Zernike aberrations.
    """
    def __init__(self,
                 pupil: VectorialPolarPupil | VectorialCartesianPupil,
                 name: str = "pupil_aberrations"):
        self.pupil = pupil

        super().__init__(
            name=name,
            psf_expr=None,
        )

    def eval_pupil_at(self, sin_t: torch.Tensor, sin_t_max: float) -> torch.Tensor:
        """Evaluate the pupil field at the radius `rho` = `r`.

        to query the zernike aberrations, normalize the pupil coordinate
        by sin_t_max such that it spans the unit disk [0,1].
        (the Pupil class operates in these normalized coordinates)

        """
        r = sin_t / sin_t_max
        n_zernike = len(self.pupil.zernike_coefficients)
        zernike_phase = torch.zeros_like(r)
        for i in range(n_zernike):
            n, l = index_to_nl(index=i)
            curr_coef = self.pupil.zernike_coefficients[i]
            if l != 0 and curr_coef != 0:
                print("Warning: Zernike coefficients for l != 0 are not supported in polar coordinates.")
            elif l == 0:
                zernike_phase += curr_coef * torch.tensor(zernike_nl(n=n, l=l, rho=r, phi=0.0))
        zernike = torch.exp(1j * zernike_phase).to(self.pupil.device)
        E = torch.tensor([self.pupil.e0x, self.pupil.e0y]) * zernike
        return E

    def eval_pupil_at_np(self, sin_t: np.ndarray, sin_t_max: float) -> np.ndarray:
        """Evaluate the pupil field at the radius `rho` = `r`.

        This version is implemented in numpy and is
        used to generate the ground truth PSF field.

        to query the zernike aberrations, normalize the pupil coordinate
        by sin_t_max such that it spans the unit disk [0,1].
        (the Pupil class operates in these normalized coordinates)

        """
        r = sin_t / sin_t_max
        n_zernike = len(self.pupil.zernike_coefficients)
        zernike_phase = np.zeros_like(r)
        for i in range(n_zernike):
            n, l = index_to_nl(index=i)
            curr_coef = self.pupil.zernike_coefficients[i]
            if l != 0 and curr_coef != 0:
                print("Warning: Zernike coefficients for l != 0 are not supported in polar coordinates.")
            elif l == 0:
                zernike_phase += curr_coef * zernike_nl(n=n, l=l, rho=r, phi=0.0)
        zernike = np.exp(1j * zernike_phase)
        E = np.stack((self.pupil.e0x * zernike, self.pupil.e0y * zernike), axis=0)
        return E

    def eval_PSF_at(self, kr: torch.Tensor, sin_t_max: float) -> torch.Tensor:
        psf_coord = kr
        I0x, I0y, I1x, I1y, I2x, I2y = _J_integral_vector(
            lambda sin_t: self.eval_pupil_at_np(sin_t, sin_t_max),
            sin_t_max,
            psf_coord,
            )

        return I0x, I0y, I1x, I1y, I2x, I2y

    def get_fields_as_polar(self, thetas: torch.Tensor, kxs: torch.Tensor, sin_t_max: float):
        """
        See TestCase.get_fields_as_polar().
        """
        sin_t = torch.sin(thetas)
        pupil_field, psf_field = self.get_fields_as_cartesian(
            s_x=sin_t,
            s_y=torch.tensor(0.0),
            sin_t_max=sin_t_max,
            norm_x=kxs,
            norm_y=kxs,
        )

        return pupil_field.squeeze(), psf_field

    def get_fields_as_cartesian(self,
                      s_x: torch.Tensor,
                      s_y: torch.Tensor,
                      norm_x: torch.Tensor,
                      norm_y: torch.Tensor,
                      sin_t_max: float,
                      ):
        s_xx, s_yy = torch.meshgrid(s_x, s_y, indexing='ij')

        sin_t_sq = s_xx ** 2 + s_yy ** 2
        s_valid = sin_t_sq <= sin_t_max ** 2
        sin_ts = torch.where(s_valid, torch.sqrt(sin_t_sq), 0.0)

        pupil_field = self.eval_pupil_at(sin_ts.ravel(), sin_t_max).reshape(2, *sin_ts.shape)
        pupil_field *= s_valid

        xx, yy = torch.meshgrid(norm_x, norm_y, indexing='ij')
        krr = torch.sqrt(xx ** 2 + yy ** 2)

        varphi = torch.atan2(yy, xx)

        krs, rr_indices = torch.unique(krr, return_inverse=True)
        I0x, I0y, I1x, I1y, I2x, I2y = self.eval_PSF_at(krs, sin_t_max)
        I0x = I0x[rr_indices]
        I0y = I0y[rr_indices]
        I1x = I1x[rr_indices]
        I1y = I1y[rr_indices]
        I2x = I2x[rr_indices]
        I2y = I2y[rr_indices]

        cos_phi, cos_twophi = torch.cos(varphi), torch.cos(2.0 * varphi)
        sin_phi, sin_twophi = torch.sin(varphi), torch.sin(2.0 * varphi)

        # updated expression with correct 1j factors
        psf_field = torch.stack(
            [I0x - I2x * cos_twophi - I2y * sin_twophi,
             I0y - I2x * sin_twophi + I2y * cos_twophi,
             -2j * (I1x * cos_phi + I1y * sin_phi)]
        )

        return pupil_field, psf_field



polar_gaussian = HankelCase(
    hankel_expr=lambda pupil_coord: 2.0 * torch.exp( -pupil_coord ** 2),
    hankel_expr_np=lambda pupil_coord: 2.0 * np.exp( -pupil_coord ** 2),
    psf_expr=lambda psf_coord: torch.exp(- (0.5 * psf_coord) ** 2),
    Rmax = 0.1,
    name = "polar_gaussian",
)

polar_step = HankelCase(
    hankel_expr=lambda pupil_coord: 2.0 * (pupil_coord <= 1.0),
    hankel_expr_np=lambda pupil_coord: 2.0 * (pupil_coord <= 1.0),
    psf_expr=lambda psf_coord: 2.0 * torch.where(psf_coord > 1e-6, bessel_j1(psf_coord) / psf_coord, 0.5 - psf_coord ** 2 / 16),
    Rmax = 0.6,
    name = "polar_step",
)

polar_exp = HankelCase(
    hankel_expr = lambda pupil_coord: torch.exp(-pupil_coord),
    hankel_expr_np = lambda pupil_coord: np.exp(-pupil_coord),
    # the closed-form expression is only valid when the upper integral bound is infinite
    # psf_expr = lambda psf_coord: (1.0 + psf_coord ** 2) ** -1.5,
    psf_expr = None,
    Rmax = 0.1,
    name = "polar_exp",
)

polar_rsq = HankelCase(
    hankel_expr = lambda xi: 1.0 / (1.0 + xi ** 2),
    hankel_expr_np = lambda xi: 1.0 / (1.0 + xi ** 2),
    psf_expr = None,
    Rmax = 0.1,
    name = "polar_rsq",
)

polar_logstep = HankelCase(
    hankel_expr = lambda pupil_coord: 4.0 * (pupil_coord <= 1.0) * \
                    torch.where(pupil_coord > 1e-6,
                                torch.log(1.0 / pupil_coord),
                                0.0),
    hankel_expr_np = lambda pupil_coord: 4.0 * (pupil_coord <= 1.0) * \
                    np.where(pupil_coord > 1e-6,
                                np.log(1.0 / pupil_coord),
                                0.0),
    psf_expr = lambda psf_coord: 4.0 * torch.where(psf_coord > 1e-6,
                    (1.0 - bessel_j0(psf_coord)) / psf_coord ** 2,
                    0.25 - psf_coord ** 2 / 64),
    Rmax = 0.6,
    name = "polar_logstep",
)


def _J_integral_scalar(e_inf: Callable, max_pupil_coord: float, psf_coord_: torch.Tensor) -> torch.Tensor:
    """
    Computes the PSF field using high-order quadrature. This function is used to generate
    ground truth/reference data against which the propagators are compared.

    Inputs:
    - e_inf: Callable. The pupil field function $e_inf(\sin{\theta})$. The provided function should
            be directly parameterized by sin_theta.
    - max_pupil_coord: float. The maximum value of the pupil coordinate, == sin(theta_max).
    - psf_coord: torch.Tensor[n_psf,]. The list of radial positions/coordinates at which to evaluate the PSF field.

    Outputs:
    - PSF_field: torch.Tensor[n_psf,]. The evaluated PSF field.
    """
    psf_coord = psf_coord_.numpy().flatten()
    out = psf_coord.copy() * 0.0
    for i in range(len(out)):
        l = psf_coord[i]
        out[i] = quad(lambda xi: e_inf(xi) / np.sqrt(1.0 - xi ** 2) * sp_j0(xi * l) * xi, a=0, b=max_pupil_coord)[0]

    return torch.tensor(out).reshape(psf_coord_.shape)


def _J_integral_vector(e_inf: Callable, sin_t_max: float, psf_coord_: torch.Tensor) -> torch.Tensor:
    """
    Computes the PSF field using high-order quadrature. This function is used to generate
    ground truth/reference data against which the propagators are compared.

    Inputs:
        - e_inf: Callable. The pupil field function $e_inf(\sin{\theta})$. The provided function should
                be directly parameterized by sin_theta.
        - sin_t_max: float. The maximum value of the pupil coordinate, == sin(theta_max).
    - psf_coord: torch.Tensor[n_psf,]. The list of radial positions/coordinates at which to evaluate the PSF field.

    Outputs:
    - I_{0/1/2}_{x/y}: torch.Tensor[n_psf,]. The evaluated Bessel function integrals; to
                be assembled into the actual PSF field.
    """
    psf_coord = psf_coord_.numpy().flatten()
    out = np.zeros((psf_coord.size, 3, 2), dtype=np.complex128)

    def integrand(xi, l):
        inv_sqrt = 1.0 / np.sqrt(1.0 - xi ** 2)
        J0 = e_inf(xi) * xi * (1.0 + inv_sqrt) * sp_j0(xi * l)
        J1 = e_inf(xi) *  (xi ** 2 * inv_sqrt) * sp_j1(xi * l)
        J2 = e_inf(xi) * xi * (1.0 - inv_sqrt) * sp_j2(xi * l)
        return np.concatenate((J0, J1, J2))   # [J0x, J0y, J1x, J1y, J2x, J2y]

    for i in range(len(out)):
        l = psf_coord[i]

        J_real = quad_vec(f=lambda xi: integrand(xi, l).real, a=0, b=sin_t_max)[0]
        J_imag = quad_vec(f=lambda xi: integrand(xi, l).imag, a=0, b=sin_t_max)[0]
        out[i,:,:].real = J_real.reshape(3,2)
        out[i,:,:].imag = J_imag.reshape(3,2)

    # output shape: [**psf_coord_shape, J_{0,1,2}, {x,y}]
    out = torch.tensor(out).reshape(*psf_coord_.shape, 3, 2)

    I0x = out[:,0,0]
    I0y = out[:,0,1]
    I1x = out[:,1,0]
    I1y = out[:,1,1]
    I2x = out[:,2,0]
    I2y = out[:,2,1]

    return I0x, I0y, I1x, I1y, I2x, I2y


'''
Tester static classes for the scalar propagators. They are used to compute PSF approximation errors
and generate error convergence plots.
'''
class ScalarPolarTester:
    @staticmethod
    def eval_error(
        N: int,
        test_case: TestCase,
        plot: bool=False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate the approximation error of the computed PSF field using an input test case. This
        method can be called on one of the analytic test cases implemented in this module
        (`HankelCase`) or an aberrated pupil (`ScalarPupilCase`).

        Inputs:
        - N: int. Grid size for the propagator.
        - test_case: TestCase. One of the implemented test cases in this module,
            e.g. `polar_step`, `polar_gaussian`.
        - plot: bool. Visualize approximation errors with plots. Default: False

        Outputs:
        - err: torch.Tensor[1,]. The scalar approximation error.
        - E_ref: torch.Tensor[N,N]. The reference PSF field.
        - E_num: torch.Tensor[N,N]. The calculated PSF field.
        """

        _pupil = ScalarPolarPupil(N)
        prop  = ScalarPolarPropagator(
            pupil=_pupil,
            n_pix_psf=2 * (N - 1) + 1,
            n_defocus=1,
            defocus_min=0,
            defocus_max=0,
            fov=1e4,
            envelope=None,
            apod_factor=False,
            gibson_lanni=False)

        far_fields, E_ref = test_case.get_fields_as_polar(
            thetas=prop.thetas,
            krs=prop.k * prop.rs,
            sin_t_max=torch.sin(prop.thetas.max()).item())

        E_ref = E_ref[prop.rr_indices]
        E_num = prop._compute_PSF_for_far_field(far_fields).squeeze()
        err = (E_ref - E_num).abs()

        if plot:
            _plot_field_comparison(E_num, E_ref, err)

        return err, E_ref, E_num

    @staticmethod
    def plot_convergence(test_case: TestCase, ord: int=1, Ns: list[int]=None) -> None:
        """
        Generate the error convergence plot for a given test case.

        Inputs:
        - test_case_data: Callable function. One of the implemented test cases in this module,
            e.g. `polar_step`, `polar_gaussian`.
        - ord: int. Error norm order. For example, `ord == 2` calculates the L2 error (root-mean
            -squared) between the numeric and exact field values. Default: 1
        - Ns: list[int]. List of grid sizes to query for the propagator. If no value is specified,
            a default set of logspaced values from 2**3 to 2**8 is used.
        """
        _plot_convergence(
            lambda N: ScalarPolarTester.eval_error(N, test_case)[0],
            Ns,
            ord,
            label="Polar",
            method_order=4,
            )
        plt.title(f"Error convergence plot: {test_case.get_name()}")


class ScalarCartesianTester:
    @staticmethod
    def eval_error(
        N: int,
        test_case: TestCase,
        plot: bool=False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate the approximation error of the computed PSF field using an input test case. This
        method can be called on one of the analytic test cases implemented in this module
        (`HankelCase`) or an aberrated pupil (`ScalarPupilCase`).

        Inputs:
        - N: int. Grid size for the propagator.
        - test_case: TestCase. One of the implemented test cases in this module,
            e.g. `polar_step`, `polar_gaussian`.
        - plot: bool. Visualize approximation errors with plots. Default: False

        Outputs:
        - err: torch.Tensor[1,]. The scalar approximation error.
        - E_ref: torch.Tensor[N,N]. The reference PSF field.
        - E_num: torch.Tensor[N,N]. The calculated PSF field.
        """

        _pupil = ScalarCartesianPupil(2 * (N - 1) + 1)
        prop = ScalarCartesianPropagator(
            pupil=_pupil,
            n_pix_psf=_pupil.n_pix_pupil,
            n_defocus=1,
            defocus_min=0,
            defocus_max=0,
            fov=1e4,
            envelope=None,
            sz_correction=True,
            apod_factor=False,
            gibson_lanni=False)

        far_fields, E_ref = test_case.get_fields_as_cartesian(
                                s_x=prop.s_x * prop.s_max,
                                s_y=prop.s_x * prop.s_max,
                                norm_x=prop.x * 2 * torch.pi / prop.s_max,
                                norm_y=prop.x * 2 * torch.pi / prop.s_max,
                                sin_t_max=prop.s_max,
                                )

        E_num = prop._compute_PSF_for_far_field(far_fields).squeeze()

        # TODO: error metric
        err = (E_ref - E_num).abs()

        if plot:
            _plot_field_comparison(E_num, E_ref, err)

        return err, E_ref, E_num

    @staticmethod
    def plot_convergence(test_case: TestCase, ord: int=1, Ns: list[int]=None) -> None:
        """
        Generate the error convergence plot for a given test case.

        Inputs:
        - test_case_data: Callable function. One of the implemented test cases in this module,
            e.g. `polar_step`, `polar_gaussian`.
        - ord: int. Error norm order. For example, `ord == 2` calculates the L2 error (root-mean
            -squared) between the numeric and exact field values. Default: 1
        - Ns: list[int]. List of grid sizes to query for the propagator. If no value is specified,
            a default set of logspaced values from 9 to 257 is used.
        """
        _plot_convergence(
            lambda N: ScalarCartesianTester.eval_error(N, test_case)[0],
            Ns,
            ord,
            label="Cartesian",
            method_order=2,
            )
        plt.title(f"Error convergence plot: {test_case.get_name()}")


class VectorPolarTester:
    @staticmethod
    def eval_error(
        N: int,
        pupil: VectorialPolarPupil,
        plot: bool=False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate the approximation error of the computed PSF field using an aberrated pupil.

        Inputs:
        - N: int. Grid size for the propagator.
        - pupil: VectorialPolarPupil. A vectorial pupil (possibly with aberrations).
        - plot: bool. Visualize approximation errors with plots. Default: False

        Outputs:
        - err: torch.Tensor[1,]. The scalar approximation error.
        - E_ref: torch.Tensor[N,N]. The reference PSF field.
        - E_num: torch.Tensor[N,N]. The calculated PSF field.
        """
        pupil_ = VectorialPolarPupil(
            e0x=pupil.e0x,
            e0y=pupil.e0y,
            n_pix_pupil=N,
            device=pupil.device,
            zernike_coefficients=pupil.zernike_coefficients,
        )
        test_case = VectorPupilCase(pupil_)
        prop  = VectorialPolarPropagator(
            pupil=pupil_,
            n_pix_psf=2 * (N - 1) + 1,
            n_defocus=1,
            defocus_min=0,
            defocus_max=0,
            fov=1e4,
            envelope=None,
            apod_factor=False,
            gibson_lanni=False)

        far_fields, E_ref = test_case.get_fields_as_polar(
            thetas=prop.thetas,
            kxs=prop.k * torch.linspace(-prop.fov / 2.0, prop.fov / 2.0, prop.n_pix_psf),
            sin_t_max=torch.sin(prop.thetas.max()).item())

        E_ref /= np.sqrt(prop.refractive_index)
        E_num = prop._compute_PSF_for_far_field(far_fields).squeeze()

        err = (E_ref - E_num).abs()

        if plot:
            field_ids = {0: 'x', 1: 'y', 2: 'z'}
            for (field_id, field_name) in field_ids.items():
                _plot_field_comparison(E_num[field_id], E_ref[field_id], err[field_id])
                plt.suptitle(rf"$E_{field_name}$")

        return err, E_ref, E_num

    @staticmethod
    def plot_convergence(pupil: VectorialPolarPupil, ord: int=1, Ns: list[int]=None) -> None:
        """
        Generate the error convergence plot for an aberrated pupil.

        Inputs:
        - pupil: VectorialPolarPupil. A vectorial pupil (possibly with aberrations).
        - ord: int. Error norm order. For example, `ord == 2` calculates the L2 error (root-mean
            -squared) between the numeric and exact field values. Default: 1
        - Ns: list[int]. List of grid sizes to query for the propagator. If no value is specified,
            a default set of logspaced values from 2**3 to 2**8 is used.
        """
        _plot_convergence(
            lambda N: VectorPolarTester.eval_error(N, pupil)[0],
            Ns,
            ord,
            label="Polar",
            method_order=4,
            )
        plt.title(f"Error convergence plot: Vector Zernike aberrations")


class VectorCartesianTester:
    @staticmethod
    def eval_error(
        N: int,
        pupil: VectorialCartesianPupil,
        plot: bool=False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate the approximation error of the computed PSF field using an aberrated pupil.

        Inputs:
        - N: int. Grid size for the propagator.
        - pupil: VectorialCartesianPupil. A vectorial pupil (possibly with aberrations).
        - plot: bool. Visualize approximation errors with plots. Default: False

        Outputs:
        - err: torch.Tensor[1,]. The scalar approximation error.
        - E_ref: torch.Tensor[N,N]. The reference PSF field.
        - E_num: torch.Tensor[N,N]. The calculated PSF field.
        """
        Ngrid = 2 * (N - 1) + 1
        pupil_ = VectorialCartesianPupil(
            e0x=pupil.e0x,
            e0y=pupil.e0y,
            n_pix_pupil=Ngrid,
            device=pupil.device,
            zernike_coefficients=pupil.zernike_coefficients)

        prop  = VectorialCartesianPropagator(
            pupil=pupil_,
            n_pix_psf=Ngrid,
            n_defocus=1,
            defocus_min=0,
            defocus_max=0,
            fov=1e4,
            sz_correction=True,
            envelope=None,
            apod_factor=False,
            gibson_lanni=False)
        far_fields = pupil_.field

        test_case = VectorPupilCase(
            VectorialPolarPupil(
                e0x=pupil.e0x,
                e0y=pupil.e0y,
                n_pix_pupil=Ngrid,
                device=pupil.device,
                zernike_coefficients=pupil.zernike_coefficients))

        _, E_ref = test_case.get_fields_as_cartesian(
                    s_x=prop.s_x * prop.s_max,
                    s_y=prop.s_x * prop.s_max,
                    norm_x=prop.x * 2 * torch.pi / prop.s_max,
                    norm_y=prop.x * 2 * torch.pi / prop.s_max,
                    sin_t_max=prop.s_max)

        E_ref /= np.sqrt(prop.refractive_index)
        E_num = prop._compute_PSF_for_far_field(far_fields).squeeze()

        # TODO: error metric
        err = (E_ref - E_num).abs()
        # err = (E_ref.abs() - E_num.abs()).abs()

        if plot:
            field_ids = {0: 'x', 1: 'y', 2: 'z'}
            for (field_id, field_name) in field_ids.items():
                _plot_field_comparison(E_num[field_id], E_ref[field_id], err[field_id])
                plt.suptitle(rf"$E_{field_name}$")

        return err, E_ref, E_num

    @staticmethod
    def plot_convergence(pupil: VectorialCartesianPupil, ord: int=1, Ns: list[int]=None) -> None:
        """
        Generate the error convergence plot for an aberrated pupil.

        Inputs:
        - pupil: VectorialCartesianPupil. A vectorial pupil (possibly with aberrations).
        - ord: int. Error norm order. For example, `ord == 2` calculates the L2 error (root-mean
            -squared) between the numeric and exact field values. Default: 1
        - Ns: list[int]. List of grid sizes to query for the propagator. If no value is specified,
            a default set of logspaced values from 2**3 to 2**8 is used.
        """
        _plot_convergence(
            lambda N: VectorCartesianTester.eval_error(N, pupil)[0],
            Ns,
            ord,
            label="Cartesian",
            method_order=2,
            )
        plt.title(f"Error convergence plot: Vector Zernike aberrations")


def _error_norm(err: torch.Tensor, ord: int=1) -> torch.Tensor:
    """
    Compute the length/norm of an error vector.
    """
    if ord == 1:
        return err.abs().mean().item()
    elif ord == 2:
        return ((err ** 2).sum() / len(err)).sqrt().item()
    elif ord == torch.inf:
        return err.abs().max().item()


def _plot_field_comparison(E: torch.Tensor, E_ref: torch.Tensor, E_err: torch.Tensor = None) -> None:
    """
    Helper function to compare the analytic PSF E field with its numerical approximation.
    """
    if E_err is None:
        E_err = (E - E_ref).abs()

    plt.figure(figsize=(8,6.4))
    plt.subplot(221)
    plt.imshow(E.abs())
    plt.title("Numeric")
    plt.colorbar()

    plt.subplot(222)
    plt.title("Exact")
    plt.imshow(E_ref.abs())
    plt.colorbar()

    plt.subplot(223)
    plt.title(f"$L_1$: {_error_norm(E_err, 1):.2e},\n$L_2$: {_error_norm(E_err, 2):.2e},\n$L_\infty$: {_error_norm(E_err, torch.inf):.2e}")
    plt.imshow(E_err)
    plt.colorbar()

    plt.subplot(224)
    N = E.shape[0]
    y = E[N//2].abs()
    y_ref = E_ref[N//2].abs()
    plt.semilogy((y - y_ref).abs())
    plt.title(f"Avg. magnitude error: {(y - y_ref).abs().mean().item():.3e}")
    plt.tight_layout()


def _plot_convergence(error_vector_getter: Callable, Ns: list[int]=None, ord: int=1, label: str=None, method_order: int=2) -> None:
    """
    Helper function to generate the error convergence plot.
    """
    if Ns is None:
        Ns = torch.unique(torch.logspace(2, 8, steps=20, base=2).to(torch.int32))
        Ns = 2 * (Ns // 2) + 1
    else:
        Ns = torch.tensor(Ns)

    if label is None:
        label = "Method"

    errs = []
    for N in tqdm(Ns):
        err_vec = error_vector_getter(int(N))
        err = _error_norm(err_vec, ord)
        errs.append(err)

    plt.loglog(Ns, errs, label=label, linewidth=2.0, zorder=3)
    plt.loglog(Ns, errs[0] * (Ns / Ns[0]) ** (-method_order), 'k--', linewidth=0.75, label=rf"$O(h^{method_order})$")
    plt.legend()
    plt.xlabel("Grid size")
    plt.ylabel("Mean abs. error")
    plt.title(f"Error convergence plot: {error_vector_getter.__name__}")
    plt.grid("on")
    plt.ylim([1e-8, 1e0])
    plt.tight_layout()
