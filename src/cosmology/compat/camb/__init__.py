"""The Cosmology API compatability library for :mod:`camb`.

This library provides wrappers for CAMB cosmology objects to be compatible
with the Cosmology API. The available wrappers are:

- :class:`.CAMBCosmology`: the Cosmology API wrapper for
  :mod:`astropy.camb.Cosmology`.


There are the following required objects for a Cosmology-API compatible library:

- constants: a module of constants. See
  :mod:`cosmology.compat.camb.constants` for details.
"""

from cosmology.compat.camb import constants
from cosmology.compat.camb.background import CAMBBackgroundCosmology
from cosmology.compat.camb.core import CAMBCosmology
from cosmology.compat.camb.standard import CAMBStandardCosmology

__all__ = [
    # Cosmology API
    "constants",
    # Wrappers
    "CAMBCosmology",
    "CAMBBackgroundCosmology",
    "CAMBStandardCosmology",
]
