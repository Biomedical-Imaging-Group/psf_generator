# Copyright Biomedical Imaging Group, EPFL 2025

"""
Coherent imaging of a nanoparticle: interferometric scattering (iSCAT), coherent bright-field (COBRI) and
dark-field microscopy.
"""
import math

import torch

from .modality import Modality, build_imager
from .particle import Particle
from ..utils.parameters import decode_complex, encode_complex, validate_number

#: Imaging schemes of :class:`ScatteringMicroscope`.
SCHEMES = ('iscat', 'cobri', 'darkfield')


class ScatteringMicroscope(Modality):
    r"""
    Wide-field coherent imaging of a Rayleigh scatterer with a plane-wave illumination.

    The sample is the usual three-layer structure of the propagators and imagers: the particle sits at a height
    :math:`z_p` above the coverslip (index :math:`n_g`) in the sample medium (:math:`n_s`), and is imaged through
    the coverslip and the immersion medium (:math:`n_i`) by an objective focused at the depth ``z_focus``. A plane
    wave of amplitude :math:`E^0 = (e_{0x}, e_{0y})` illuminates it at normal incidence and induces the dipole
    :math:`\mathbf{p} \propto \alpha \, t_{\mathrm{ill}} E^0`, whose image :math:`\mathbf{E}^{\mathrm{sca}}` is
    computed by a dipole imager (:mod:`psf_generator.imaging`). The camera records

    .. math:: I(\boldsymbol{\rho}) = |\mathbf{E}^{\mathrm{ref}} + \mathbf{E}^{\mathrm{sca}}(\boldsymbol{\rho})|^2,

    with a reference wave :math:`\mathbf{E}^{\mathrm{ref}}` that depends on the scheme:

    - ``'iscat'``: the illumination comes through the objective; the reference is the part reflected at the
      coverslip-sample interface, :math:`\mathbf{E}^{\mathrm{ref}} = \beta \, r_{gs} \, E^0`
      (:math:`r_{gs} = (n_g - n_s)/(n_g + n_s)` at normal incidence, :math:`\beta` an optional attenuation), and
      the particle is illuminated by the transmitted part :math:`t_{gs} E^0`.
    - ``'cobri'``: the illumination comes from the sample side; the reference is the transmitted illumination
      :math:`\beta \, t_{sg} E^0` and the particle sees the incident field :math:`E^0` directly.
    - ``'darkfield'``: like iSCAT with the reference blocked (:math:`\beta = 0`), so
      :math:`I = |\mathbf{E}^{\mathrm{sca}}|^2`.

    The interference depends on the optical path difference between the reference and the scattered light. On
    top of the path :math:`\Lambda(\theta)` from the particle to the objective included in the imager, the
    illumination reaches the particle after the path :math:`\Delta + n_s z_p` (iSCAT, dark-field) or
    :math:`-n_s z_p` (COBRI), and the reference travels :math:`2\Delta` (iSCAT) or :math:`\Delta` (COBRI), where
    :math:`\Delta = (n_i t_i + n_g t_g) - (n_i^* t_i^* + n_g^* t_g^*)` is the excess single-pass path through the
    immersion medium and the coverslip with respect to the design conditions. On the optical axis the phase of
    the scattered light relative to the reference is therefore :math:`2 k n_s z_p` in iSCAT and :math:`0` in
    COBRI, which reproduces the fast oscillation of the iSCAT signal with the particle height and the
    :math:`\xi = \pm 1` term of Dong et al. 2021.

    Units and normalization: lengths are in nanometer, the incident amplitude is :math:`|E^0| = 1` and the
    intensities are expressed in units of the incident intensity; a factor :math:`(n_i / n_d) / M^2` common to
    the reference and the scattered light (magnification :math:`M`, camera in a medium :math:`n_d`) is dropped,
    see :class:`psf_generator.imaging.DipoleImager`. The particle is a Rayleigh scatterer (:class:`Particle`);
    multiple reflections in the coverslip, the reflection of the illumination at the immersion-coverslip interface
    and the action of the attenuation mask on the scattered light are neglected.

    Parameters
    ----------
    particle : Particle
        The scatterer.
    scheme : str, optional
        `'iscat'` (default), `'cobri'` or `'darkfield'`.
    attenuation : float, optional
        Amplitude attenuation :math:`\beta \geq 0` of the reference wave (1 is no attenuation, 0 is dark-field).
        Default value is `1.0`.
    e0x, e0y : complex, optional
        Components of the incident plane wave. Default is an x-polarized wave of unit amplitude.
    imager : str, dict or DipoleImager, optional
        The dipole imager computing the image of the induced dipole: `'spherical'` (default, fast) or
        `'cartesian'` (any pupil aberration), a parameter dictionary or an instance.
    **imager_kwargs
        Constructor arguments of the imager when it is given by name: `wavelength`, `na`, `pix_size`,
        `n_pix_psf`, `n_pix_pupil`, `z_focus`, the refractive indices and thicknesses of the layers, `device`, ...

    """

    #: Scheme imposed by the subclasses (None for the generic class).
    _fixed_scheme = None

    def __init__(self, particle: Particle, scheme: str = 'iscat', attenuation: float = 1.0,
                 e0x=1.0, e0y=0.0, imager='spherical', **imager_kwargs):
        if isinstance(particle, dict):
            particle = Particle.from_dict(particle)
        if not isinstance(particle, Particle):
            raise TypeError(f'particle must be a Particle, not {type(particle)}.')
        if self._fixed_scheme is not None and scheme != self._fixed_scheme:
            raise ValueError(f'{type(self).__name__} is a {self._fixed_scheme!r} microscope, got scheme={scheme!r}.')
        if scheme not in SCHEMES:
            raise ValueError(f'Unknown scheme {scheme!r}, choose from {SCHEMES}.')
        validate_number('attenuation', attenuation, 0)
        self.particle = particle
        self.scheme = scheme
        self.attenuation = float(attenuation)
        self.e0x = complex(e0x)
        self.e0y = complex(e0y)
        self.imager = build_imager(imager, imager_kwargs)

    @classmethod
    def get_name(cls) -> str:
        return cls._fixed_scheme or 'scattering'

    # ------------------------------------------------------------------------------------------------------------
    # Shortcuts to the imager
    # ------------------------------------------------------------------------------------------------------------
    @property
    def wavelength(self) -> float:
        """Wavelength in nanometer."""
        return self.imager.wavelength

    @property
    def k(self) -> float:
        """Vacuum wavenumber in 1/nm."""
        return self.imager.k

    @property
    def n_s(self) -> float:
        """Refractive index of the sample medium."""
        return self.imager.n_s

    @property
    def n_g(self) -> float:
        """Refractive index of the coverslip."""
        return self.imager.n_g

    @property
    def n_i(self) -> float:
        """Refractive index of the immersion medium."""
        return self.imager.n_i

    @property
    def x(self) -> torch.Tensor:
        """Lateral coordinates of the image grid in nanometer, see ``DipoleImager.x``."""
        return self.imager.x

    @property
    def device(self):
        """Computational backend of the imager."""
        return self.imager.device

    # ------------------------------------------------------------------------------------------------------------
    # Physics
    # ------------------------------------------------------------------------------------------------------------
    def polarizability(self) -> complex:
        """Polarizability of the particle in the sample medium, in nm\\ :sup:`3`."""
        return self.particle.polarizability(self.n_s)

    def excess_path(self) -> float:
        r"""
        Excess optical path :math:`\Delta` of a single on-axis pass through the immersion medium and the coverslip
        with respect to the design conditions, in nanometer.
        """
        imager = self.imager
        return ((imager.n_i * imager.t_i + imager.n_g * imager.t_g)
                - (imager.n_i0 * imager.t_i0 + imager.n_g0 * imager.t_g0))

    def _normal_incidence(self) -> dict:
        """Fresnel amplitude coefficients at normal incidence between the layers (i: immersion, g: glass, s: sample)."""
        n_s, n_g, n_i = self.n_s, self.n_g, self.n_i
        return {
            't_ig': 2.0 * n_i / (n_i + n_g),
            't_gi': 2.0 * n_g / (n_g + n_i),
            't_gs': 2.0 * n_g / (n_g + n_s),
            't_sg': 2.0 * n_s / (n_s + n_g),
            'r_gs': (n_g - n_s) / (n_g + n_s),
        }

    def _incident_vector(self) -> torch.Tensor:
        return torch.tensor([self.e0x, self.e0y, 0.0], dtype=torch.complex128)

    def reference_field(self) -> torch.Tensor:
        r"""
        Reference wave at the camera, in object-space units.

        Returns
        -------
        field : torch.Tensor
            Complex vector of shape `(3,)` (uniform over the image), zero for dark-field.

        """
        coefficients = self._normal_incidence()
        if self.scheme == 'iscat':
            amplitude = coefficients['t_ig'] * coefficients['r_gs'] * coefficients['t_gi']
            phase = 2.0 * self.k * self.excess_path()
        elif self.scheme == 'cobri':
            amplitude = coefficients['t_sg'] * coefficients['t_gi']
            phase = self.k * self.excess_path()
        else:
            amplitude, phase = 0.0, 0.0
        amplitude = self.attenuation * amplitude * complex(math.cos(phase), math.sin(phase))
        return (amplitude * self._incident_vector()).to(torch.complex64).to(self.device)

    @property
    def reference_intensity(self) -> float:
        """Intensity of the reference wave (the background of the image), in units of the incident intensity."""
        return float((self.reference_field().abs() ** 2).sum())

    def scattered_field(self, positions=(0.0, 0.0, 0.0)) -> torch.Tensor:
        r"""
        Field scattered by the particle at the camera, in object-space units.

        Parameters
        ----------
        positions : array-like of shape `(3,)` or `(n_positions, 3)`, optional
            Positions :math:`(x_p, y_p, z_p)` of the particle in nanometer (height :math:`z_p` above the coverslip).

        Returns
        -------
        field : torch.Tensor
            Complex field of shape `(n_positions, 3, n_pix_psf, n_pix_psf)`.

        """
        positions = self.imager._as_positions(positions)
        z_p = positions[:, 2]
        coefficients = self._normal_incidence()
        if self.scheme == 'cobri':
            illumination = 1.0
            phase = -self.k * self.n_s * z_p
        else:
            illumination = coefficients['t_ig'] * coefficients['t_gs']
            phase = self.k * (self.excess_path() + self.n_s * z_p)
        k_s = self.k * self.n_s
        amplitude = k_s ** 2 * self.polarizability() * illumination / (4.0 * math.pi)
        field = self.imager.compute_image(amplitude * self._incident_vector(), positions)
        phase_factor = torch.exp(1j * phase).to(torch.complex64).to(field.device)
        return field * phase_factor[:, None, None, None]

    def compute_fields(self, positions=(0.0, 0.0, 0.0)):
        """Reference and scattered fields, see :meth:`reference_field` and :meth:`scattered_field`."""
        return self.reference_field(), self.scattered_field(positions)

    def compute_image(self, positions=(0.0, 0.0, 0.0)) -> torch.Tensor:
        r"""
        Intensity recorded by the camera, :math:`|\mathbf{E}^{\mathrm{ref}} + \mathbf{E}^{\mathrm{sca}}|^2`.

        Parameters
        ----------
        positions : array-like of shape `(3,)` or `(n_positions, 3)`, optional
            Positions :math:`(x_p, y_p, z_p)` of the particle in nanometer.

        Returns
        -------
        image : torch.Tensor
            Real intensity of shape `(n_positions, n_pix_psf, n_pix_psf)`, in units of the incident intensity.

        """
        reference, scattered = self.compute_fields(positions)
        total = reference[None, :, None, None] + scattered
        return (total.abs() ** 2).sum(dim=1)

    def compute_contrast(self, positions=(0.0, 0.0, 0.0)) -> torch.Tensor:
        r"""
        Interferometric contrast :math:`(I - I^{\mathrm{ref}}) / I^{\mathrm{ref}}`, the interferometric PSF (iPSF).

        Parameters
        ----------
        positions : array-like of shape `(3,)` or `(n_positions, 3)`, optional
            Positions :math:`(x_p, y_p, z_p)` of the particle in nanometer.

        Returns
        -------
        contrast : torch.Tensor
            Real contrast of shape `(n_positions, n_pix_psf, n_pix_psf)`.

        """
        reference_intensity = self.reference_intensity
        if reference_intensity == 0.0:
            raise ValueError('The contrast is not defined without a reference wave (dark-field, or attenuation=0); '
                             'use compute_image instead.')
        return (self.compute_image(positions) - reference_intensity) / reference_intensity

    # ------------------------------------------------------------------------------------------------------------
    # Parameters
    # ------------------------------------------------------------------------------------------------------------
    def _get_args(self) -> dict:
        args = {'particle': self.particle.to_dict()}
        if self._fixed_scheme is None:
            args['scheme'] = self.scheme
        args.update({
            'attenuation': self.attenuation,
            'e0x': encode_complex(self.e0x),
            'e0y': encode_complex(self.e0y),
            'imager': self.imager.to_dict(),
        })
        return args

    @classmethod
    def _decode_args(cls, args: dict) -> dict:
        args = super()._decode_args(args)
        if cls._fixed_scheme is not None:
            scheme = args.pop('scheme', cls._fixed_scheme)
            if scheme != cls._fixed_scheme:
                raise ValueError(f'The parameters describe a {scheme!r} microscope, not {cls._fixed_scheme!r}.')
        if isinstance(args.get('particle'), dict):
            args['particle'] = Particle.from_dict(args['particle'])
        for key in ('e0x', 'e0y'):
            if key in args:
                args[key] = decode_complex(args[key])
        return args


class ISCATMicroscope(ScatteringMicroscope):
    """
    Interferometric scattering microscope (iSCAT): the light scattered by the particle interferes with the
    illumination reflected at the coverslip-sample interface. See :class:`ScatteringMicroscope`.
    """

    _fixed_scheme = 'iscat'

    def __init__(self, particle: Particle, attenuation: float = 1.0, e0x=1.0, e0y=0.0, imager='spherical',
                 **imager_kwargs):
        super().__init__(particle, scheme='iscat', attenuation=attenuation, e0x=e0x, e0y=e0y, imager=imager,
                         **imager_kwargs)


class COBRIMicroscope(ScatteringMicroscope):
    """
    Coherent bright-field microscope (COBRI): the light scattered by the particle interferes with the transmitted
    illumination. See :class:`ScatteringMicroscope`.
    """

    _fixed_scheme = 'cobri'

    def __init__(self, particle: Particle, attenuation: float = 1.0, e0x=1.0, e0y=0.0, imager='spherical',
                 **imager_kwargs):
        super().__init__(particle, scheme='cobri', attenuation=attenuation, e0x=e0x, e0y=e0y, imager=imager,
                         **imager_kwargs)


class DarkFieldMicroscope(ScatteringMicroscope):
    """
    Dark-field microscope: only the light scattered by the particle reaches the camera (the iSCAT geometry with
    the reference blocked). See :class:`ScatteringMicroscope`.
    """

    _fixed_scheme = 'darkfield'

    def __init__(self, particle: Particle, e0x=1.0, e0y=0.0, imager='spherical', **imager_kwargs):
        super().__init__(particle, scheme='darkfield', attenuation=0.0, e0x=e0x, e0y=e0y, imager=imager,
                         **imager_kwargs)

    @classmethod
    def _decode_args(cls, args: dict) -> dict:
        args = super()._decode_args(args)
        args.pop('attenuation', None)
        return args
