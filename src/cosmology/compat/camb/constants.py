"""Astropy cosmology constants.

Note that :mod:`astropy` constants have astropy units.

From the :mod:`cosmology.api`, the list of required constants is:

- G: Gravitational constant G in pc km2 s-2 Msol-1.
- speed_of_light: Speed of light in km s-1.
"""

from camb.constants import c as c_ms

__all__ = ["c", "G"]

c = c_ms / 1e3  # [km s-1]
G = 4.30091727003628e-3  # [pc km2 s-2 Msol-1]  CODATA 2018 value
