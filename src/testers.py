'''
This file contains several classes for measuring the accuracy and convergence of the scalar propagators.
'''
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Tuple, List
from tqdm import tqdm
from torch.special import bessel_j0, bessel_j1
from scipy.special import j0 as sp_j0
from scipy.integrate import quad
from .pupil import ScalarPolarPupil, ScalarCartesianPupil
from .propagator import ScalarPolarPropagator, ScalarCartesianPropagator


def eval_as_cartesian(func: Callable, 
                      s_x: torch.Tensor, 
                      s_y: torch.Tensor, 
                      s_max: float, 
                      x: torch.Tensor, 
                      y: torch.Tensor, 
                      k: float, 
                      Rmax: float, 
                      z: float = None) -> Tuple[torch.Tensor, torch.Tensor]:
    '''
    Evaluate the input test case `func` on a Cartesian grid.

    Inputs:
    - func: Callable. One of the test cases implemented in this module, e.g. `polar_gaussian`.
    - s_x, s_y: torch.Tensor[n_pix_pupil,]. The pupil coordinates where the pupil far-field
                is evaluated.
    - s_max: float. The maximum input direction for the pupil far-field (== sin(theta_max)).
    - x, y: torch.Tensor[n_pix_pupil,]. The PSF Cartesian coordinates where the PSF is evaluated.
    - k: float. The propagator wavenumber.    
    - Rmax: float. A dimensionless quantity between [0,1]. Defines the spread/variance of the 
                pupil far-field.
    - z: float. The propagator defocus.

    Outputs:
    - pupil_field: torch.Tensor[n_pix_pupil,]. The evaluated pupil field. This is
                    used as input for the propagator's numerical integration routine.
    - psf_field: torch.Tensosr[n_pix_psf,]. The evaluated PSF field.
    '''

    # for the FFT-based propagator, the input (pupil) and output (PSF) grids are
    # constrained to be of the same size. 
    assert (s_x.shape == s_y.shape)
    assert (x.shape == y.shape)
    assert (s_x.shape == x.shape)

    s_xx, s_yy = torch.meshgrid(s_x, s_y, indexing='ij')

    sin_t_sq = s_xx ** 2 + s_yy ** 2
    s_valid = sin_t_sq <= s_max ** 2
    s_z = torch.where(s_valid, torch.sqrt(1.0 - sin_t_sq), 0.0)
    pupil_coord = torch.where(s_valid, torch.sqrt(sin_t_sq) / Rmax, 0.0)

    xx, yy = torch.meshgrid(x, y, indexing='ij')
    rr = torch.sqrt(xx ** 2 + yy ** 2)

    rho, rho_indices = torch.unique(rr, return_inverse=True)
    psf_coord = k * rho * Rmax

    pupil_field, psf_field = func(pupil_coord, psf_coord)
    pupil_field /= Rmax ** 2

    psf_field = psf_field[rho_indices]

    if z is not None:
        inv_defocus = torch.exp(-1j * k * s_z * z)
        return pupil_field * inv_defocus * s_z, psf_field
    else:
        return pupil_field * s_z, psf_field

def eval_as_polar(func: Callable, 
                  thetas: torch.Tensor, 
                  rho: torch.Tensor, 
                  k: float, 
                  Rmax: float, 
                  z: float = None) -> Tuple[torch.Tensor, torch.Tensor]:
    '''
    Evaluate the input test case `func` on a polar grid.

    Inputs:
    - func: Callable. One of the test cases implemented in this module, e.g. `polar_gaussian`.
    - thetas: torch.Tensor[n_pix_pupil,]. The pupil coordinates where the pupil far-field
                is evaluated.
    - rhos: torch.Tensor[n_pix_psf,]. The PSF radial coordinates where the PSF is evaluated.
    - k: float. The propagator wavenumber.    
    - Rmax: float. A dimensionless quantity between [0,1]. Defines the spread/variance of the 
                pupil far-field.
    - z: float. The propagator defocus.

    Outputs:
    - pupil_field: torch.Tensor[n_pix_pupil,]. The evaluated pupil field. This is
                    used as input for the propagator's numerical integration routine.
    - psf_field: torch.Tensor[n_pix_psf,]. The evaluated PSF field.
    '''
    sin_t = torch.sin(thetas)
    cos_t = torch.cos(thetas)
    pupil_coord = sin_t / Rmax
    psf_coord = k * rho * Rmax
    pupil_field, psf_field = func(pupil_coord, psf_coord)
    pupil_field /= Rmax ** 2

    if z is not None:
        inv_defocus = torch.exp(-1j * k * cos_t * z)
        return pupil_field * inv_defocus * cos_t, psf_field
    else:
        return pupil_field * cos_t, psf_field

'''
Function <-> Hankel transform pairs
pupil_field: ... without accounting for compensation factors
psf_field: actual psf field
pupil_coord: normalized pupil coord (sin theta / R == sqrt(s_x ** 2 + s_y ** 2) / R)
psf_coord: normalized psf coord (k * rho * R)
'''
def sp_integral(pupil_func: Callable, max_pupil_coord_: torch.Tensor, psf_coord_: torch.Tensor) -> torch.Tensor:
    '''
    Computes the PSF field using high-order quadrature. This function is used to generate
    ground truth/reference data against which the propagators are compared.
    '''
    max_pupil_coord = max_pupil_coord_.numpy().item()
    psf_coord = psf_coord_.numpy().flatten()
    out = psf_coord.copy() * 0.0
    for i in range(len(out)):
        l = psf_coord[i]
        out[i] = quad(lambda xi: pupil_func(xi) * sp_j0(xi * l) * xi, a=0, b=max_pupil_coord)[0]

    return torch.tensor(out).reshape(psf_coord_.shape)     

def polar_step(pupil_coord: torch.Tensor, psf_coord: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    pupil_field = 2.0 * (pupil_coord <= 1.0)
    psf_field = 2.0 * torch.where(psf_coord > 1e-6, bessel_j1(psf_coord) / psf_coord, 0.5 - psf_coord ** 2 / 16)
    return pupil_field, psf_field

def polar_exp(pupil_coord: torch.Tensor, psf_coord: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    pupil_func = lambda xi: torch.exp(-xi)
    pupil_func_np = lambda xi: np.exp(-xi)

    pupil_field = pupil_func(pupil_coord)
    psf_field = sp_integral(pupil_func_np, pupil_coord.max(), psf_coord)
    # the closed-form expression is only valid when the upper integral bound is infinite
    # psf_field = (1.0 + psf_coord ** 2) ** -1.5
    return pupil_field, psf_field

def polar_rsq(pupil_coord: torch.Tensor, psf_coord: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    pupil_func = lambda xi: 1.0 / (1.0 + xi ** 2)

    pupil_field = pupil_func(pupil_coord)
    psf_field = sp_integral(pupil_func, pupil_coord.max(), psf_coord)
    return pupil_field, psf_field

def polar_gaussian(pupil_coord: torch.Tensor, psf_coord: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    pupil_field = 2.0 * torch.exp( -pupil_coord ** 2)
    psf_field = torch.exp(- (0.5 * psf_coord) ** 2)
    return pupil_field, psf_field

def polar_logstep(pupil_coord: torch.Tensor, psf_coord: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    pupil_field = 4.0 * (pupil_coord <= 1.0) * \
                torch.where(pupil_coord > 1e-6,
                            torch.log(1.0 / pupil_coord),
                            0.0)
    psf_field = 4.0 * torch.where(psf_coord > 1e-6, 
                    (1.0 - bessel_j0(psf_coord)) / psf_coord ** 2, 
                    0.25 - psf_coord ** 2 / 64)
    return pupil_field, psf_field

def polar_pupil(pupil: ScalarPolarPupil, pupil_coord, psf_coord, sin_t_max) -> Tuple[torch.Tensor, torch.Tensor]:
    sin_t_max = float(sin_t_max)
    pupil_field = pupil.eval_field_at(pupil_coord.ravel() / sin_t_max)#.squeeze()       # == pupil.field.squeeze()
    pupil_field = pupil_field.reshape(pupil_coord.shape).squeeze()

    psf_field = torch.zeros_like(psf_coord, dtype=torch.complex64)
    # Note: xi == sin(theta)
    psf_field.real = sp_integral(
            lambda xi: pupil.eval_field_at_np(xi / sin_t_max).real / np.cos(np.arcsin(xi)),
            torch.tensor(sin_t_max),
            psf_coord,
            )

    psf_field.imag = sp_integral(
            lambda xi: pupil.eval_field_at_np(xi / sin_t_max).imag / np.cos(np.arcsin(xi)),
            torch.tensor(sin_t_max),
            psf_coord,
            )

    return pupil_field, psf_field


'''
Tester static classes for the scalar propagators. They are used to compute PSF approximation errors 
and generate error convergence plots.
'''

class ScalarPolarTester:
    @staticmethod
    def eval_error_for_testcase(
        N: int, 
        test_case_data: Callable, 
        plot: bool=False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        '''
        Compute the average approximation error of the PSF field for the chosen test case.

        Inputs:
        - N: int. Grid size for the pupil & propagator.
        - test_case_data: Callable function. One of the implemented test cases in this module, 
            e.g. `polar_step`, `polar_gaussian`.
        - ord: int. Error norm order. For example, `ord == 2` calculates the L2 error (root-mean
            -squared) between the numeric and exact field values. Default: 1

        Outputs:
        - err: torch.Tensor[1,]. The scalar approximation error.
        - E_ref: torch.Tensor[N,N]. The reference PSF field.
        - E_num: torch.Tensor[N,N]. The calculated PSF field.
        '''
        _pupil = ScalarPolarPupil(N)
        prop  = ScalarPolarPropagator(
            pupil=_pupil,
            n_pix_psf=_pupil.n_pix_pupil,
            n_defocus=1, 
            defocus_min=0,
            defocus_max=0, 
            fov=1e4,
            envelope=None, 
            apod_factor=False, 
            gibson_lanni=False)

        Rmax = 0.1  # between 0.0 and 1.0
        far_fields, E_ref = eval_as_polar(test_case_data, thetas=prop.thetas, rho=prop.rs, k=prop.k, Rmax=Rmax)
        E_ref = E_ref[prop.rr_indices]
        E_num = prop._compute_PSF_for_far_field(far_fields).squeeze()
        err = (E_ref - E_num).abs()

        if plot:
            _plot_field_comparison(E_num, E_ref, err)

        return err, E_ref, E_num
    
    @staticmethod
    def eval_error_for_pupil(
        pupil: ScalarPolarPupil,
        plot: bool=False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        '''
        Compute the average approximation error of the PSF field generated by the input
        pupil and its Zernike aberrations.

        Inputs:
        - pupil: ScalarPolarPupil. Input pupil with aberrations.
        - test_case_data: Callable function. One of the implemented test cases in this module, 
            e.g. `polar_step`, `polar_gaussian`.
        - ord: int. Error norm order. For example, `ord == 2` calculates the L2 error (root-mean
            -squared) between the numeric and exact field values. Default: 1

        Outputs:
        - err: torch.Tensor[1,]. The scalar approximation error.
        - E_ref: torch.Tensor[N,N]. The reference PSF field.
        - E_num: torch.Tensor[N,N]. The calculated PSF field.
        '''
        prop = ScalarPolarPropagator(
            pupil=pupil,
            n_pix_psf=2 * (pupil.n_pix_pupil - 1) + 1,
            n_defocus=1, 
            defocus_min=0,
            defocus_max=0, 
            fov=1e4,
            envelope=None, 
            apod_factor=False, 
            gibson_lanni=False)
        
        sin_t = torch.sin(prop.thetas)
        pupil_field, E_ref = polar_pupil(pupil, sin_t, prop.rs * prop.k, sin_t.max())
        E_ref = E_ref[prop.rr_indices]
        E_num = prop._compute_PSF_for_far_field(pupil_field).squeeze()

        err = (E_ref - E_num).abs()

        if plot:
            _plot_field_comparison(E_num, E_ref, err)

        return err, E_ref, E_num


    @staticmethod
    def plot_convergence_for_testcase(test_case_data: Callable, ord: int=1, Ns: List[int]=None) -> None:
        '''
        Generate the error convergence plot for a given test case.

        Inputs:
        - test_case_data: Callable function. One of the implemented test cases in this module, 
            e.g. `polar_step`, `polar_gaussian`.
        - ord: int. Error norm order. For example, `ord == 2` calculates the L2 error (root-mean
            -squared) between the numeric and exact field values. Default: 1
        - Ns: List[int]. List of grid sizes to query for the propagator. If no value is specified,
            a default set of logspaced values from 2**3 to 2**8 is used.
        '''
        _plot_convergence(
            lambda N: ScalarPolarTester.eval_error_for_testcase(N, test_case_data)[0], 
            Ns, 
            ord,
            label="Polar",
            )
        plt.title(f"Error convergence plot: {test_case_data.__name__}")

    @staticmethod
    def plot_convergence_for_pupil(pupil: ScalarPolarPupil, ord: int=1, Ns: List[int]=None) -> None:
        '''
        '''
        get_resampled_pupil = lambda N: ScalarPolarPupil(
            n_pix_pupil=N, 
            zernike_coefficients=pupil.zernike_coefficients)

        _plot_convergence(
            lambda N: ScalarPolarTester.eval_error_for_pupil(get_resampled_pupil(N))[0], 
            Ns, 
            ord,
            label="Polar",
            )
        plt.title("Error convergence plot: ScalarPolarProp + ScalarPolarPupil")


class ScalarCartesianTester:
    @staticmethod
    def eval_error_for_testcase(
        N: int, 
        test_case_data: Callable, 
        plot: bool=False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        '''
        Evaluate the average approximation error over the PSF field. This method can either be 
        called on an existing scalar Cartesian propagator, or for an input grid size `N`. In the
        latter case, the propagator is initialized internally.

        Inputs:
        - N: int. Grid size for the propagator.
        - test_case_data: Callable function. One of the implemented test cases in this module, 
            e.g. `polar_step`, `polar_gaussian`.
        - ord: int. Error norm order. For example, `ord == 2` calculates the L2 error (root-mean
            -squared) between the numeric and exact field values. Default: 1
        - prop: ScalarCartesianPropagator. The propagator for which to compute the approximation error.

        Outputs:
        - err: torch.Tensor[1,]. The scalar approximation error.
        - E_ref: torch.Tensor[N,N]. The reference PSF field.
        - E_num: torch.Tensor[N,N]. The calculated PSF field.
        '''
        
        _pupil = ScalarCartesianPupil(N)
        prop  = ScalarCartesianPropagator(
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
        
        Rmax = 0.1  # between 0.0 and 1.0
        # note the extra scale factors for s_x, x
        far_fields, E_ref = eval_as_cartesian(test_case_data, 
                                                s_x=prop.s_x * prop.s_max,
                                                s_y=prop.s_x * prop.s_max,
                                                s_max=prop.s_max,
                                                x=prop.x / prop.k * 2 * torch.pi / prop.s_max,
                                                y=prop.x / prop.k * 2 * torch.pi / prop.s_max,
                                                k=prop.k, 
                                                Rmax=Rmax)

        E_num = prop._compute_PSF_for_far_field(far_fields).squeeze()
        
        err = (E_ref - E_num).abs()
        # err = (E_ref.abs() - E_num.abs()).abs()

        if plot:
            _plot_field_comparison(E_num, E_ref, err)

        return err, E_ref, E_num
    
    def eval_error_for_pupil(
        pupil: ScalarPolarPupil,
        plot: bool=False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        '''
        Compute the average approximation error of the PSF field generated by the input
        pupil and its Zernike aberrations.

        Inputs:
        - pupil: ScalarPolarPupil. Input pupil with aberrations.
        - test_case_data: Callable function. One of the implemented test cases in this module, 
            e.g. `polar_step`, `polar_gaussian`.
        - ord: int. Error norm order. For example, `ord == 2` calculates the L2 error (root-mean
            -squared) between the numeric and exact field values. Default: 1

        Outputs:
        - err: torch.Tensor[1,]. The scalar approximation error.
        - E_ref: torch.Tensor[N,N]. The reference PSF field.
        - E_num: torch.Tensor[N,N]. The calculated PSF field.
        '''

        # convert the polar pupil into the equivalent cartesian pupil
        pupil_c = ScalarCartesianPupil(
            n_pix_pupil=1 + 2*(pupil.n_pix_pupil - 1),
            zernike_coefficients=pupil.zernike_coefficients,
        )
        
        prop = ScalarCartesianPropagator(
            pupil=pupil_c,
            n_pix_psf=pupil_c.n_pix_pupil,
            n_defocus=1, 
            defocus_min=0,
            defocus_max=0, 
            fov=1e4,
            envelope=None, 
            sz_correction=True,
            apod_factor=False, 
            gibson_lanni=False)

        pupil_field = pupil_c.field.squeeze()

        x = prop.x * 2 * torch.pi / prop.s_max
        xx, yy = torch.meshgrid(x, x, indexing='ij')
        rr = torch.sqrt(xx ** 2 + yy ** 2)
        psf_coord, rr_indices = torch.unique(rr, return_inverse=True)
        E_ref = polar_pupil(pupil, torch.tensor(0.0), psf_coord, prop.s_max)[1]
        E_ref = E_ref[rr_indices]

        E_num = prop._compute_PSF_for_far_field(pupil_field).squeeze()
        err = (E_ref - E_num).abs()

        if plot:
            _plot_field_comparison(E_num, E_ref, err)

        return err, E_ref, E_num
    
    @staticmethod
    def plot_convergence_for_testcase(test_case_data: Callable, ord: int=1, Ns: List[int]=None) -> None:
        '''
        Generate the error convergence plot for a given test case.

        Inputs:
        - test_case_data: Callable function. One of the implemented test cases in this module, 
            e.g. `polar_step`, `polar_gaussian`.
        - ord: int. Error norm order. For example, `ord == 2` calculates the L2 error (root-mean
            -squared) between the numeric and exact field values. Default: 1
        - Ns: List[int]. List of grid sizes to query for the propagator. If no value is specified,
            a default set of logspaced values from 9 to 257 is used.
        '''
        _plot_convergence(
            lambda N: ScalarCartesianTester.eval_error_for_testcase(N, test_case_data)[0], 
            Ns, 
            ord,
            label="Cartesian",
            )
        plt.title(f"Error convergence plot: {test_case_data.__name__}")


    @staticmethod
    def plot_convergence_for_pupil(pupil: ScalarPolarPupil, ord: int=1, Ns: List[int]=None) -> None:
        '''
        '''
        get_resampled_pupil = lambda N: ScalarPolarPupil(
            n_pix_pupil=N, 
            zernike_coefficients=pupil.zernike_coefficients)
        
        _plot_convergence(
            lambda N: ScalarCartesianTester.eval_error_for_pupil(get_resampled_pupil(N))[0],
            Ns, 
            ord,
            label="Cartesian",
            )
        plt.title("Error convergence plot: ScalarCartesianProp + ScalarPolarPupil")


def _error_norm(err: torch.Tensor, ord: int=1) -> torch.Tensor:
    '''
    Compute the length/norm of an error vector.
    '''
    if ord == 1:
        return err.abs().mean().item()
    elif ord == 2:
        return ((err ** 2).sum() / len(err)).sqrt().item()
    elif ord == torch.inf:
        return err.abs().max().item()

def _plot_field_comparison(E: torch.Tensor, E_ref: torch.Tensor, E_err: torch.Tensor = None) -> None:
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

def _plot_convergence(error_vector_getter: Callable, Ns: List[int]=None, ord: int=1, label: str=None) -> None:
    if Ns is None:
        Ns = torch.unique(torch.logspace(3, 8, steps=20, base=2).to(torch.int32))
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
    plt.loglog(Ns, errs[0] * (Ns / Ns[0]) ** (-4), 'k--', linewidth=0.75, label=rf"$O(h^{4})$")
    plt.legend()
    plt.xlabel("Grid size")
    plt.ylabel("Mean abs. error")
    plt.title(f"Error convergence plot: {error_vector_getter.__name__}")
    plt.grid("on")
    plt.ylim([1e-8, 1e0])
    plt.tight_layout()
