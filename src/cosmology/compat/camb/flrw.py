"""The Cosmology API compatability wrapper for CAMB."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from cosmology.api import FLRWAPIConformantWrapper
from cosmology.compat.camb import constants
from cosmology.compat.camb.core import CAMBCosmology
from typing_extensions import TypeAlias

if TYPE_CHECKING:
    from numpy import floating
    from numpy.typing import NDArray

    NDFloating: TypeAlias = NDArray[floating[Any]]


__all__: list[str] = []


_MPCGYR_KMS = np.array("9.7309928209912e35", dtype=np.float64)  # [Mpc / km]


@dataclass(frozen=True)
class CAMBFLRW(CAMBCosmology, FLRWAPIConformantWrapper):
    """Cosmology API Protocol for FLRW-like cosmologies.

    This is a protocol class that defines the standard API for FLRW-like
    cosmologies. It is not intended to be instantiaed. Instead, it should be
    used for ``isinstance`` checks or as an ABC for libraries that wish to
    define a compatible cosmology class.
    """

    @property
    def H0(self) -> NDFloating:
        """Hubble constant at z=0 in km s-1 Mpc-1."""
        return np.asarray(self._params.H0)

    @property
    def Om0(self) -> NDFloating:
        """Matter density at z=0."""
        return np.asarray(self._params.omegam)

    @property
    def Ode0(self) -> NDFloating:
        """Dark energy density at z=0."""
        return np.asarray(self.cosmo.omega_de)

    @property
    def Tcmb0(self) -> NDFloating:
        """Temperature of the CMB at z=0."""
        return np.asarray(self._params.TCMB)

    @property
    def Neff(self) -> NDFloating:
        """Effective number of neutrino species."""
        return np.asarray(self._params.N_eff)

    @property
    def m_nu(self) -> tuple[NDFloating, ...]:
        """Mass of neutrinos."""
        raise NotImplementedError  # TODO!

    @property
    def Ob0(self) -> NDFloating:
        """Baryon density at z=0."""
        return np.asarray(self._params.ombh2 / self.h**2)

    # ==============================================================

    @property
    def scale_factor0(self) -> NDFloating:
        """Scale factor at z=0."""
        return np.asarray(1.0)

    # ----------------------------------------------
    # Hubble

    @property
    def h(self) -> NDFloating:
        r"""Dimensionless Hubble parameter, h = H_0 / (100 [km/sec/Mpc])."""
        return np.asarray(self._params.H0 / 100)

    @property
    def hubble_distance(self) -> NDFloating:
        """Hubble distance in Mpc."""
        return constants.speed_of_light / self.H0

    @property
    def hubble_time(self) -> NDFloating:
        """Hubble time in Gyr."""
        return _MPCGYR_KMS / self.H0

    # ----------------------------------------------
    # Omega

    @property
    def Otot0(self) -> NDFloating:
        r"""Omega total; the total density/critical density at z=0.

        .. math::

            \Omega_{\rm tot} = \Omega_{\rm m} + \Omega_{\rm \gamma} +
            \Omega_{\rm \nu} + \Omega_{\rm de} + \Omega_{\rm k}
        """
        raise NotImplementedError  # TODO!

    @property
    def Odm0(self) -> NDFloating:
        """Omega dark matter; dark matter density/critical density at z=0."""
        raise NotImplementedError  # TODO!

    @property
    def Ok0(self) -> NDFloating:
        """Omega curvature; the effective curvature density/critical density at z=0."""
        return np.asarray(self._params.omk)

    @property
    def Ogamma0(self) -> NDFloating:
        """Omega gamma; the density/critical density of photons at z=0."""
        raise NotImplementedError  # TODO!

    @property
    def Onu0(self) -> NDFloating:
        """Omega nu; the density/critical density of neutrinos at z=0."""
        raise NotImplementedError  # TODO!

    # ----------------------------------------------
    # Density

    @property
    def rho_critical0(self) -> NDFloating:
        """Critical density at z = 0 in Msol Mpc-3."""
        # H0 is in km s-1 Mpc-1
        # G is in pc Msol-1 (km/s)^2
        return 3e6 * self.H0**2 / (8 * np.pi * constants.G)

    @property
    def rho_tot0(self) -> NDFloating:
        """Total density at z = 0 in Msol Mpc-3."""
        raise NotImplementedError  # TODO!

    @property
    def rho_m0(self) -> NDFloating:
        """Matter density at z = 0 in Msol Mpc-3."""
        return self.rho_critical0 * self.Om0

    @property
    def rho_de0(self) -> NDFloating:
        """Dark energy density at z = 0 in Msol Mpc-3."""
        return self.rho_critical0 * self.Ode0

    @property
    def rho_b0(self) -> NDFloating:
        """Baryon density at z = 0 in Msol Mpc-3."""
        return self.rho_critical0 * self.Ob0

    @property
    def rho_dm0(self) -> NDFloating:
        """Dark matter density at z = 0 in Msol Mpc-3."""
        return self.rho_critical0 * self.Odm0

    @property
    def rho_k0(self) -> NDFloating:
        """Curvature density at z = 0 in Msol Mpc-3."""
        return self.rho_critical0 * self.Ok0

    @property
    def rho_gamma0(self) -> NDFloating:
        """Radiation density at z = 0 in Msol Mpc-3."""
        return self.rho_critical0 * self.Ogamma0

    @property
    def rho_nu0(self) -> NDFloating:
        """Neutrino density at z = 0 in Msol Mpc-3."""
        return self.rho_critical0 * self.Onu0

    # ==============================================================
    # Methods

    def scale_factor(self, z: NDFloating, /) -> NDFloating:
        """Redshift-dependenct scale factor :math:`a = a_0 / (1 + z)`."""
        return np.asarray(self.scale_factor0 / (z + 1))

    # ----------------------------------------------
    # Hubble

    def H(self, z: NDFloating, /) -> NDFloating:
        """Hubble function :math:`H(z)` in km s-1 Mpc-1."""  # noqa: D402
        return np.asarray(self.cosmo.hubble_parameter(z))

    def efunc(self, z: NDFloating, /) -> NDFloating:
        """Standardised Hubble function :math:`E(z) = H(z)/H_0`."""
        return self.H(z) / self.H0

    def inv_efunc(self, z: NDFloating, /) -> NDFloating:
        """Inverse of ``efunc``."""
        return self.H0 / self.H(z)

    # ----------------------------------------------
    # Omega

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
            vars=["total"],
            format="array",
        )[:, 0]

    def Om(self, z: NDFloating, /) -> NDFloating:
        """Redshift-dependent non-relativistic matter density parameter.

        Notes
        -----
        This does not include neutrinos, even if non-relativistic at the
        redshift of interest; see `Onu`.
        """
        return np.asarray(
            self.cosmo.get_Omega("cdm", z) + self.cosmo.get_Omega("baryon", z),
        )

    def Ob(self, z: NDFloating, /) -> NDFloating:
        """Redshift-dependent baryon density parameter.

        Raises
        ------
        ValueError
            If ``Ob0`` is `None`.
        """
        return np.asarray(self.cosmo.get_Omega("baryon", z))

    def Odm(self, z: NDFloating, /) -> NDFloating:
        """Redshift-dependent dark matter density parameter.

        Raises
        ------
        ValueError
            If ``Ob0`` is `None`.

        Notes
        -----
        This does not include neutrinos, even if non-relativistic at the
        redshift of interest.
        """
        return np.asarray(self.cosmo.get_Omega("cdm", z))

    def Ok(self, z: NDFloating, /) -> NDFloating:
        """Redshift-dependent curvature density parameter."""
        return np.asarray(self.cosmo.get_Omega("K", z))

    def Ode(self, z: NDFloating, /) -> NDFloating:
        """Redshift-dependent dark energy density parameter."""
        ...

    def Ogamma(self, z: NDFloating, /) -> NDFloating:
        """Redshift-dependent photon density parameter."""
        ...

    def Onu(self, z: NDFloating, /) -> NDFloating:
        r"""Redshift-dependent neutrino density parameter.

        The energy density of neutrinos relative to the critical density at each
        redshift. Note that this includes their kinetic energy (if
        they have mass), so it is not equal to the commonly used :math:`\sum
        \frac{m_{\nu}}{94 eV}`, which does not include kinetic energy.
        Returns `float` if the input is scalar.
        """
        ...

    # ----------------------------------------------
    # Rho

    def rho_critical(self, z: NDFloating, /) -> NDFloating:
        """Redshift-dependent critical density in Msol Mpc-3."""
        ...

    def rho_tot(self, z: NDFloating, /) -> NDFloating:
        """Redshift-dependent total density in Msol Mpc-3."""
        ...

    def rho_m(self, z: NDFloating, /) -> NDFloating:
        """Redshift-dependent matter density in Msol Mpc-3."""
        ...

    def rho_de(self, z: NDFloating, /) -> NDFloating:
        """Redshift-dependent dark energy density in Msol Mpc-3."""
        ...

    def rho_k(self, z: NDFloating, /) -> NDFloating:
        """Redshift-dependent curvature density in Msol Mpc-3."""
        ...

    def rho_gamma(self, z: NDFloating, /) -> NDFloating:
        """Redshift-dependent photon density in Msol Mpc-3."""
        ...

    def rho_nu(self, z: NDFloating, /) -> NDFloating:
        """Redshift-dependent neutrino density in Msol Mpc-3."""
        ...

    def rho_b(self, z: NDFloating, /) -> NDFloating:
        """Redshift-dependent baryon density in Msol Mpc-3."""
        ...

    def rho_dm(self, z: NDFloating, /) -> NDFloating:
        """Redshift-dependent dark matter density in Msol Mpc-3."""
        ...

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
        raise NotImplementedError  # TODO

    def comoving_volume(self, z: NDFloating, /) -> NDFloating:
        r"""Comoving volume in cubic Mpc.

        This is the volume of the universe encompassed by redshifts less than
        ``z``. For the case of :math:`\Omega_k = 0` it is a sphere of radius
        `comoving_distance` but it is less intuitive if :math:`\Omega_k` is not.
        """
        raise NotImplementedError  # TODO

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
        raise NotImplementedError  # TODO

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
