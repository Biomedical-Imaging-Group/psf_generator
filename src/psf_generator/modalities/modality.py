# Copyright Biomedical Imaging Group, EPFL 2025

"""
The abstract modality: the complete image-formation model of a microscopy technique.

A modality composes the building blocks of the library -- the propagators for the illumination path, the dipole
imagers for the detection path and a model of the sample -- into the image recorded by a given technique.
"""
from abc import ABC, abstractmethod

import torch

from ..imaging import IMAGERS, DipoleImager
from ..utils.parameters import Parametrized


def build_imager(imager, imager_kwargs: dict) -> DipoleImager:
    """
    Build the dipole imager of a modality.

    Parameters
    ----------
    imager : str, dict or DipoleImager
        Name of an imager (`'spherical'` or `'cartesian'`), the dictionary returned by ``DipoleImager.to_dict``
        or an imager instance.
    imager_kwargs : dict
        Constructor arguments of the imager; only allowed with a name.

    """
    if isinstance(imager, DipoleImager):
        if imager_kwargs:
            raise ValueError(f'The imager is already built: the extra arguments {sorted(imager_kwargs)} cannot '
                             f'be applied to it.')
        return imager
    if isinstance(imager, dict):
        if imager_kwargs:
            raise ValueError(f'The imager is given as a parameter dictionary: the extra arguments '
                             f'{sorted(imager_kwargs)} cannot be applied to it.')
        return DipoleImager.from_dict(imager)
    if isinstance(imager, str):
        if imager not in IMAGERS:
            raise ValueError(f'Unknown imager {imager!r}, choose from {sorted(IMAGERS)} or pass a DipoleImager.')
        return IMAGERS[imager](**imager_kwargs)
    raise TypeError(f'imager must be a name, a parameter dictionary or a DipoleImager, not {type(imager)}.')


class Modality(Parametrized, ABC):
    """
    Base class of the modalities.

    A modality describes how a microscopy technique forms an image: illumination of the sample, response of the
    sample (e.g. the dipole induced in a nanoparticle) and detection of the light. Every modality computes the
    recorded intensity with :meth:`compute_image` and can be saved and restored like the propagators
    (``to_dict``, ``from_dict``, ``save_parameters``, ``load_parameters``).

    """

    #: Key under which the name of the modality is stored by :meth:`to_dict`.
    _registry_key = 'modality'

    @classmethod
    def _registry(cls) -> dict:
        from . import MODALITIES
        return MODALITIES

    @abstractmethod
    def compute_image(self, positions=(0.0, 0.0, 0.0)) -> torch.Tensor:
        """
        Compute the intensity recorded by the camera for the given positions of the emitter or scatterer.

        Parameters
        ----------
        positions : array-like of shape `(3,)` or `(n_positions, 3)`, optional
            Positions :math:`(x_p, y_p, z_p)` in nanometer, see ``DipoleImager.compute_image``.

        Returns
        -------
        image : torch.Tensor
            Real intensity of shape `(n_positions, n_pix_psf, n_pix_psf)`.

        """
        raise NotImplementedError
