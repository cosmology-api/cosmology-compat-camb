"""The Cosmology API compatability wrapper for CAMB."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cosmology.api import BackgroundCosmologyWrapperAPI
from cosmology.compat.camb import constants
from cosmology.compat.camb.core import CAMBCosmology, NDFloating

__all__: list[str] = []


##############################################################################


@dataclass(frozen=True)
class CAMBBackgroundCosmology(CAMBCosmology, BackgroundCosmologyWrapperAPI[NDFloating]):
    """FLRW Cosmology API wrapper for CAMB cosmologies."""

    @property
    def scale_factor0(self) -> NDFloating:
        """Scale factor at z=0."""
        return np.asarray(1.0)

    @property
    def Otot0(self) -> NDFloating:
        r"""Omega total; the total density/critical density at z=0.

        .. math::

            \Omega_{\rm tot} = \Omega_{\rm m} + \Omega_{\rm \gamma} +
            \Omega_{\rm \nu} + \Omega_{\rm de} + \Omega_{\rm k}
        """
        return (  # TODO: this is a hack, but it works for now.
            self.cosmo.get_background_densities(1, ["tot"], format="array")[:, 0]
            / self.critical_density0
        )

    @property
    def critical_density0(self) -> NDFloating:
        """Critical density at z = 0 in Msol Mpc-3."""
        return 3e6 * self.H0**2 / (8 * np.pi * constants.G)

    # ==============================================================
    # Methods

    def scale_factor(self, z: NDFloating, /) -> NDFloating:
        """Redshift-dependenct scale factor :math:`a = a_0 / (1 + z)`."""
        return np.asarray(self.scale_factor0 / (z + 1))

    def Otot(self, z: NDFloating, /) -> NDFloating:
        r"""Redshift-dependent total density parameter.

        This is the sum of the matter, radiation, neutrino, dark energy, and
        curvature density parameters.

        .. math::

            \Omega_{\rm tot} = \Omega_{\rm m} + \Omega_{\rm \gamma} +
            \Omega_{\rm \nu} + \Omega_{\rm de} + \Omega_{\rm k}
        """
        return self.cosmo.get_background_densities(
            self.scale_factor(z),
            vars=["tot"],
            format="array",
        )[:, 0] / (8 * np.pi * constants.G * self.scale_factor(z) ** 4)

    def critical_density(self, z: NDFloating, /) -> NDFloating:
        """Redshift-dependent critical density in Msol Mpc-3."""
        a = self.scale_factor(z)
        return self.cosmo.get_background_densities(a, vars=["tot"], format="array")[
            :,
            0,
        ] / (8 * np.pi * constants.G * a**4)

    # ----------------------------------------------
    # Time

    def age(self, z: NDFloating, /) -> NDFloating:
        """Age of the universe in Gyr at redshift ``z``."""
        return np.asarray(self.cosmo.physical_time(z))

    def lookback_time(self, z: NDFloating, /) -> NDFloating:
        """Lookback time to redshift ``z`` in Gyr.

        The lookback time is the difference between the age of the Universe now
        and the age at redshift ``z``.
        """
        # TODO: cache the age of the universe
        return np.asarray(self.cosmo.physical_time(0) - self.cosmo.physical_time(z))

    # ----------------------------------------------
    # Comoving distance

    def comoving_distance(self, z: NDFloating, /) -> NDFloating:
        r"""Comoving line-of-sight distance :math:`d_c(z)` in Mpc.

        The comoving distance along the line-of-sight between two objects
        remains constant with time for objects in the Hubble flow.
        """
        # TODO: have a way to set the tolerance
        return np.asarray(self.cosmo.comoving_radial_distance(z, tol=0.0001))

    def comoving_transverse_distance(self, z: NDFloating, /) -> NDFloating:
        r"""Transverse comoving distance :math:`d_M(z)` in Mpc.

        This value is the transverse comoving distance at redshift ``z``
        corresponding to an angular separation of 1 radian. This is the same as
        the comoving distance if :math:`\Omega_k` is zero (as in the current
        concordance Lambda-CDM model).
        """
        return self.angular_diameter_distance(z) * (z + 1)

    def comoving_volume(self, z: NDFloating, /) -> NDFloating:
        r"""Comoving volume in cubic Mpc.

        This is the volume of the universe encompassed by redshifts less than
        ``z``. For the case of :math:`\Omega_k = 0` it is a sphere of radius
        `comoving_distance` but it is less intuitive if :math:`\Omega_k` is not.
        """
        if self.Ok0 == 0:
            return 4.0 / 3.0 * np.pi * self.comoving_distance(z) ** 3

        dh = self.hubble_distance
        x = self.comoving_transverse_distance(z) / dh
        term1 = 4.0 * np.pi * dh**3 / (2.0 * self.Ok0)
        term2 = x * np.sqrt(1 + self.Ok0 * (x) ** 2)
        term3 = np.sqrt(np.abs(self.Ok0)) * x

        if self.Ok0 > 0:
            return term1 * (term2 - 1.0 / np.sqrt(np.abs(self.Ok0)) * np.arcsinh(term3))
        else:
            return term1 * (term2 - 1.0 / np.sqrt(np.abs(self.Ok0)) * np.arcsin(term3))

    def differential_comoving_volume(self, z: NDFloating, /) -> NDFloating:
        r"""Differential comoving volume in cubic Mpc per steradian.

        If :math:`V_c` is the comoving volume of a redshift slice with solid
        angle :math:`\Omega`, this function ...

        .. math::

            \mathtt{dvc(z)}
            = \frac{1}{d_H^3} \, \frac{dV_c}{d\Omega \, dz}
            = \frac{x_M^2(z)}{E(z)}
            = \frac{\mathtt{xm(z)^2}}{\mathtt{ef(z)}} \;.

        """
        return (
            self.comoving_transverse_distance(z) / self.hubble_distance
        ) ** 2 / self.efunc(z)

    # ----------------------------------------------
    # Angular diameter distance

    def angular_diameter_distance(self, z: NDFloating, /) -> NDFloating:
        """Angular diameter distance :math:`d_A(z)` in Mpc.

        This gives the proper (sometimes called 'physical') transverse
        distance corresponding to an angle of 1 radian for an object
        at redshift ``z`` ([1]_, [2]_, [3]_).

        References
        ----------
        .. [1] Weinberg, 1972, pp 420-424; Weedman, 1986, pp 421-424.
        .. [2] Weedman, D. (1986). Quasar astronomy, pp 65-67.
        .. [3] Peebles, P. (1993). Principles of Physical Cosmology, pp 325-327.
        """
        return np.asarray(self.cosmo.angular_diameter_distance(z))

    # ----------------------------------------------
    # Luminosity distance

    def luminosity_distance(self, z: NDFloating, /) -> NDFloating:
        """Redshift-dependent luminosity distance in Mpc.

        This is the distance to use when converting between the bolometric flux
        from an object at redshift ``z`` and its bolometric luminosity [1]_.

        References
        ----------
        .. [1] Weinberg, 1972, pp 420-424; Weedman, 1986, pp 60-62.
        """
        return self.cosmo.luminosity_distance(z)
