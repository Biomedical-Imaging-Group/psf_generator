# Copyright Biomedical Imaging Group, EPFL 2025

"""
Save and restore the parameters of a model as JSON.

Every model of the library (the propagators, the dipole imagers and the modalities) derives from
:class:`Parametrized`: it is fully described by its constructor arguments, which :meth:`Parametrized.to_dict`
returns as a JSON-serialisable dictionary and :meth:`Parametrized.from_dict` turns back into an identical model.
Each family of models keeps a registry of its concrete classes (``PROPAGATORS``, ``IMAGERS`` and ``MODALITIES``)
so that a saved file can be loaded without knowing the exact class in advance.

"""
import inspect
import json
import numbers
import os
from abc import ABC, abstractmethod

import torch

__all__ = ['Parametrized', 'encode_complex', 'decode_complex', 'validate_device', 'validate_number']


def encode_complex(value) -> list:
    """Encode a (possibly complex) number as a JSON-friendly ``[real, imag]`` pair."""
    value = complex(value)
    return [value.real, value.imag]


def decode_complex(value) -> complex:
    """Inverse of :func:`encode_complex`; also accepts the ``str(complex)`` form written by versions <= 0.1.0."""
    if isinstance(value, (list, tuple)):
        return complex(value[0], value[1])
    if isinstance(value, str):
        return complex(value.replace(' ', ''))
    return complex(value)


def validate_device(device) -> None:
    """Raise a :class:`ValueError` if `device` is not a valid PyTorch device specification."""
    try:
        torch.device(device)
    except (RuntimeError, TypeError) as error:
        raise ValueError(f"Invalid device {device!r}: {error}. Valid examples are 'cpu', 'cuda', 'cuda:0' "
                         f"and 'mps'.") from error


def validate_number(name: str, value, minimum, strict: bool = False) -> None:
    """Raise a :class:`ValueError` unless `value` is a number above `minimum` (excluded if `strict`).

    ``numbers.Real`` rather than ``(int, float)``, so that the NumPy scalars a caller may get from an array
    or from ``np.arange`` (``np.int64`` is not a subclass of ``int``) are accepted; ``bool`` is not a size.
    """
    valid = isinstance(value, numbers.Real) and not isinstance(value, bool)
    valid = valid and (value > minimum if strict else value >= minimum)
    if not valid:
        comparison = 'greater than' if strict else 'at least'
        raise ValueError(f'{name} must be a number {comparison} {minimum}, got {value!r}.')


class Parametrized(ABC):
    """
    Base class of every model that can be rebuilt from its constructor arguments.

    Subclasses provide a name (:meth:`get_name`), their constructor arguments as JSON values (:meth:`_get_args`)
    and, if some values need decoding, the inverse conversion (:meth:`_decode_args`). A family of models shares
    a registry (:meth:`_registry`) that maps names to classes, stored under the key ``_registry_key`` of the
    parameter dictionary.

    """

    #: Key under which the name of the model is stored in the parameter dictionary.
    _registry_key: str = 'model'

    @classmethod
    @abstractmethod
    def get_name(cls) -> str:
        """Get the name of the model, e.g. 'scalar_cartesian'."""
        raise NotImplementedError

    @classmethod
    def _registry(cls) -> dict:
        """Concrete models of this family by name."""
        return {}

    @abstractmethod
    def _get_args(self) -> dict:
        """Constructor arguments of the model as JSON-serialisable values (see :meth:`to_dict`)."""
        raise NotImplementedError

    @classmethod
    def _decode_args(cls, args: dict) -> dict:
        """Inverse of :meth:`_get_args`: turn the JSON values back into constructor arguments."""
        return dict(args)

    def to_dict(self) -> dict:
        """
        Return the parameters of the model as a JSON-serialisable dictionary.

        The dictionary holds the constructor arguments plus the name of the model (under the key
        ``'propagator'``, ``'imager'`` or ``'modality'`` depending on the family), so that :meth:`from_dict`
        can rebuild an identical model.

        Notes
        -----
        - Complex numbers (e.g. ``e0x``) are stored as ``[real, imag]`` pairs, Zernike coefficients as a list
          and the integrator of the spherical models by name.
        - Tensors that cannot be written to JSON (a custom ``special_phase_mask`` or a ``custom_field``) are not
          stored; a warning is issued.

        """
        return {self._registry_key: self.get_name(), **self._get_args()}

    @classmethod
    def from_dict(cls, parameters: dict) -> 'Parametrized':
        """
        Build a model from a dictionary produced by :meth:`to_dict` (or :meth:`save_parameters`).

        Called on an abstract base class (e.g. ``Propagator``), the class is taken from the name stored in the
        dictionary; called on a concrete class, that class is used (and the stored name, if present, must
        describe it or one of its subclasses).

        Parameters
        ----------
        parameters : dict
            Parameters as returned by :meth:`to_dict`.

        Returns
        -------
        model : Parametrized
            A new model.

        """
        key = cls._registry_key
        parameters = dict(parameters)
        name = parameters.pop(key, None)
        if name is None:
            if inspect.isabstract(cls):
                raise ValueError(f"The parameters do not name a {key}: add the {key!r} key or call from_dict on "
                                 f"a concrete class.")
            target = cls
        else:
            registry = cls._registry()
            if name not in registry:
                raise ValueError(f'Unknown {key} {name!r}, choose from {sorted(registry)}.')
            target = registry[name]
            if not issubclass(target, cls):
                own = cls.__name__ if inspect.isabstract(cls) else repr(cls.get_name())
                raise ValueError(f'The parameters describe a {name!r} {key}, not {own}.')
        return target(**target._decode_args(parameters))

    @classmethod
    def load_parameters(cls, json_filepath: str) -> 'Parametrized':
        """
        Build a model from a JSON file written by :meth:`save_parameters`.

        Parameters
        ----------
        json_filepath : str
            Path to the JSON file.

        """
        with open(json_filepath) as file:
            return cls.from_dict(json.load(file))

    def save_parameters(self, json_filepath: str) -> None:
        """
        Save the parameters of the model in a JSON file (see :meth:`to_dict`).

        Parameters
        ----------
        json_filepath : str
            Path of the JSON file to write.

        """
        directory = os.path.dirname(json_filepath)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(json_filepath, 'w') as file:
            json.dump(self.to_dict(), file, indent=2)
