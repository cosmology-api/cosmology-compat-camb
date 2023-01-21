"""Test the Cosmology API compat library."""

import camb
import pytest

from cosmology.api import CosmologyAPI, CosmologyWrapperAPI
from cosmology.compat.camb import CAMBCosmology

################################################################################
# TESTS
################################################################################


class Test_CAMBCosmology:
    @pytest.fixture(scope="class")
    def cosmo(self) -> camb.CAMBdata:
        # EXAMPLE FROM https://camb.readthedocs.io/en/latest/CAMBdemo.html

        # Set up a new set of parameters for CAMB
        pars = camb.CAMBparams()
        # This function sets up CosmoMC-like settings, with one massive neutrino
        # and helium set using BBN consistency
        pars.set_cosmology(H0=67.5, ombh2=0.022, omch2=0.122, mnu=0.06, omk=0, tau=0.06)
        pars.InitPower.set_params(As=2e-9, ns=0.965, r=0)
        pars.set_for_lmax(2500, lens_potential_accuracy=0)

        return camb.get_results(pars)

    @pytest.fixture(scope="class")
    def wrapper(self, cosmo: camb.CAMBdata) -> CAMBCosmology:
        return CAMBCosmology(cosmo)

    # =========================================================================
    # Tests

    def test_wrapper_is_compliant(self, wrapper):
        """Test that AstropyCosmology is a CosmologyWrapperAPI."""
        assert isinstance(wrapper, CosmologyAPI)
        assert isinstance(wrapper, CosmologyWrapperAPI)

    def test_getattr(self, wrapper, cosmo):
        """Test that the wrapper can access the attributes of the wrapped object."""
        # The base Cosmology API doesn't have grhocrit
        assert wrapper.grhocrit == cosmo._params.grhocrit
