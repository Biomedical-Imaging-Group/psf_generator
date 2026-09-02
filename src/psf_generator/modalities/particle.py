# Copyright Biomedical Imaging Group, EPFL 2025

"""
A Rayleigh scatterer.
"""
import math

from ..utils.parameters import decode_complex, encode_complex, validate_number

#: Mass of one dalton, in grams.
DALTON = 1.66053906660e-24


class Particle:
    r"""
    A Rayleigh scatterer: a sphere much smaller than the wavelength, described by its radius and its refractive
    index (or permittivity).

    In a medium of refractive index :math:`n_m` (permittivity :math:`\epsilon_m = n_m^2`) the particle has the
    Clausius-Mossotti polarizability

    .. math:: \alpha = 4\pi a^3 \frac{\epsilon_p - \epsilon_m}{\epsilon_p + 2\epsilon_m}

    (in nm\ :sup:`3`), so that an incident field :math:`\mathbf{E}_{\mathrm{inc}}` induces the dipole moment
    :math:`\mathbf{p} = \epsilon_0 \epsilon_m \alpha \mathbf{E}_{\mathrm{inc}}`, which radiates the far field
    :math:`\mathbf{E}(\mathbf{r}) = \frac{k_m^2 \alpha}{4\pi} [(\hat{\mathbf{s}} \times \mathbf{E}_{\mathrm{inc}})
    \times \hat{\mathbf{s}}] \, \mathrm{e}^{\mathrm{i} k_m r} / r` with :math:`k_m = 2\pi n_m / \lambda`, and has the
    scattering cross-section :math:`\sigma_{\mathrm{sca}} = k_m^4 |\alpha|^2 / (6\pi)`.

    Parameters
    ----------
    radius : float
        Radius of the particle, in nanometer.
    refractive_index : complex, optional
        Complex refractive index of the particle material at the wavelength of interest.
    permittivity : complex, optional
        Complex relative permittivity of the particle material (the square of the refractive index). Exactly one
        of `refractive_index` and `permittivity` must be given.

    Examples
    --------
    A 30 nm gold nanoparticle at 517.5 nm (Johnson & Christy):

    >>> gold = Particle(radius=15.0, permittivity=-3.7328 + 2.7725j)

    """

    def __init__(self, radius: float, refractive_index=None, permittivity=None):
        validate_number('radius', radius, 0, strict=True)
        if (refractive_index is None) == (permittivity is None):
            raise ValueError('Give exactly one of refractive_index and permittivity.')
        self.radius = float(radius)
        if refractive_index is not None:
            self.refractive_index = complex(refractive_index)
            self.permittivity = self.refractive_index ** 2
        else:
            self.refractive_index = None
            self.permittivity = complex(permittivity)

    @classmethod
    def from_mass(cls, mass: float, density: float, refractive_index=None, permittivity=None) -> 'Particle':
        """
        Build the particle of a given mass, e.g. a protein in mass photometry.

        Parameters
        ----------
        mass : float
            Mass in kilodalton.
        density : float
            Mass density of the material in g/cm\\ :sup:`3`. Proteins are commonly modelled with a specific
            volume of about 0.73 mL/g, i.e. a density of about 1.35 g/cm\\ :sup:`3`.
        refractive_index, permittivity : complex, optional
            Optical property of the material, see :class:`Particle`.

        """
        validate_number('mass', mass, 0, strict=True)
        validate_number('density', density, 0, strict=True)
        volume = mass * 1e3 * DALTON / density * 1e21  # nm^3
        radius = (3.0 * volume / (4.0 * math.pi)) ** (1.0 / 3.0)
        return cls(radius, refractive_index=refractive_index, permittivity=permittivity)

    @property
    def volume(self) -> float:
        """Volume of the particle, in nm\\ :sup:`3`."""
        return 4.0 / 3.0 * math.pi * self.radius ** 3

    def polarizability(self, n_medium: float) -> complex:
        """Clausius-Mossotti polarizability in a medium of refractive index `n_medium`, in nm\\ :sup:`3`."""
        epsilon_m = complex(n_medium) ** 2
        return 3.0 * self.volume * (self.permittivity - epsilon_m) / (self.permittivity + 2.0 * epsilon_m)

    def scattering_cross_section(self, wavelength: float, n_medium: float) -> float:
        """Scattering cross-section in nm\\ :sup:`2` at the given wavelength (nm) in a medium of index `n_medium`."""
        k_m = 2.0 * math.pi * n_medium / wavelength
        return k_m ** 4 * abs(self.polarizability(n_medium)) ** 2 / (6.0 * math.pi)

    def to_dict(self) -> dict:
        """Parameters of the particle as a JSON-serialisable dictionary (see :meth:`from_dict`)."""
        return {
            'radius': self.radius,
            'refractive_index': None if self.refractive_index is None else encode_complex(self.refractive_index),
            'permittivity': encode_complex(self.permittivity),
        }

    @classmethod
    def from_dict(cls, parameters: dict) -> 'Particle':
        """Build a particle from the dictionary returned by :meth:`to_dict`."""
        refractive_index = parameters.get('refractive_index')
        if refractive_index is not None:
            return cls(parameters['radius'], refractive_index=decode_complex(refractive_index))
        return cls(parameters['radius'], permittivity=decode_complex(parameters['permittivity']))

    def __eq__(self, other) -> bool:
        return isinstance(other, Particle) and self.to_dict() == other.to_dict()

    def __repr__(self) -> str:
        if self.refractive_index is not None:
            return f'Particle(radius={self.radius!r}, refractive_index={self.refractive_index!r})'
        return f'Particle(radius={self.radius!r}, permittivity={self.permittivity!r})'
