"""The Cosmology API compatability wrapper for CAMB."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cosmology.api import StandardCosmologyWrapperAPI
from cosmology.compat.camb import constants
from cosmology.compat.camb.background import CAMBBackgroundCosmology
from cosmology.compat.camb.core import NDFloating

__all__: list[str] = []


##############################################################################
# PARAMETERS


_MPCGYR_KMS = np.array("9.7309928209912e35", dtype=np.float64)  # [Mpc / km]


##############################################################################


@dataclass(frozen=True)
class CAMBStandardCosmology(
    CAMBBackgroundCosmology,
    StandardCosmologyWrapperAPI[NDFloating],
):
    """FLRW Cosmology API wrapper for CAMB cosmologies."""

    @property
    def Tcmb0(self) -> NDFloating:
        """Temperature of the CMB at z=0."""
        return np.asarray(self._params.TCMB)

    # ----------------------------------------------
    # Hubble

    @property
    def H0(self) -> NDFloating:
        """Hubble constant at z=0 in km s-1 Mpc-1."""
        return np.asarray(self._params.H0)

    @property
    def h(self) -> NDFloating:
        r"""Dimensionless Hubble parameter, h = H_0 / (100 [km/sec/Mpc])."""
        return np.asarray(self._params.H0 / 100)

    @property
    def hubble_distance(self) -> NDFloating:
        """Hubble distance in Mpc."""
        return constants.c / self.H0

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
        return (  # TODO: this is a hack, but it works for now.
            self.cosmo.get_background_densities(1, ["tot"], format="array")[:, 0]
            / self.critical_density0
        )

    @property
    def Om0(self) -> NDFloating:
        """Matter density at z=0."""
        return np.asarray(self._params.omegam)

    @property
    def Odm0(self) -> NDFloating:
        """Omega dark matter; dark matter density/critical density at z=0."""
        return np.asarray(self._params.omegac)

    @property
    def Ok0(self) -> NDFloating:
        """Omega curvature; the effective curvature density/critical density at z=0."""
        return np.asarray(self._params.omk)

    @property
    def Ob0(self) -> NDFloating:
        """Baryon density at z=0."""
        return np.asarray(self._params.omegab)

    @property
    def Ode0(self) -> NDFloating:
        """Dark energy density at z=0."""
        return np.asarray(self.cosmo.omega_de)

    @property
    def Ogamma0(self) -> NDFloating:
        """Omega gamma; the density/critical density of photons at z=0."""
        return np.asarray(self.cosmo.get_Omega("photon", z=0))

    @property
    def Onu0(self) -> NDFloating:
        """Omega nu; the density/critical density of neutrinos at z=0."""
        raise NotImplementedError  # TODO!

    # ----------------------------------------------
    # Density

    @property
    def critical_density0(self) -> NDFloating:
        """Critical density at z = 0 in Msol Mpc-3."""
        # H0 is in (km/s) Mpc-1; G is in pc Msol-1 (km/s)^2
        # so we pick up a factor of 1e6 to get Msol Mpc-3
        return 3e6 * self.H0**2 / (8 * np.pi * constants.G)

    # ----------------------------------------------
    # Neutrinos

    @property
    def Neff(self) -> NDFloating:
        """Effective number of neutrino species."""
        return np.asarray(self._params.N_eff)

    @property
    def m_nu(self) -> tuple[NDFloating, ...]:
        """Mass of neutrinos."""
        raise NotImplementedError  # TODO!

    # ==============================================================
    # Methods

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
        return np.asarray(self.cosmo.get_Omega("de", z))

    def Ogamma(self, z: NDFloating, /) -> NDFloating:
        """Redshift-dependent photon density parameter."""
        return np.asarray(self.cosmo.get_Omega("photon", z))

    def Onu(self, z: NDFloating, /) -> NDFloating:
        r"""Redshift-dependent neutrino density parameter."""
        return np.asarray(
            self.cosmo.get_Omega("neutrino", z) + self.cosmo.get_Omega("nu", z),
        )
