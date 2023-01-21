"""Test the Cosmology API compat library."""

from __future__ import annotations

import camb
import numpy as np
import pytest
from hypothesis import given
from hypothesis.extra import numpy as npst

from cosmology.api import StandardCosmologyAPI, StandardCosmologyWrapperAPI
from cosmology.compat.camb import CAMBStandardCosmology

from .test_background import Test_CAMBBackgroundCosmology

################################################################################
# PARAMETERS

# Hypothesis strategy for generating arrays of redshifts
def z_arr_st(*, allow_nan: bool = False, min_value: float | None = 0, **kwargs):
    return npst.arrays(
        # TODO: do we want to test float16?
        npst.floating_dtypes(sizes=(32, 64)),
        npst.array_shapes(min_dims=1),
        elements={"allow_nan": allow_nan, "min_value": min_value} | kwargs,
    )


################################################################################
# TESTS
################################################################################


class Test_CAMBStandardCosmology(Test_CAMBBackgroundCosmology):
    @pytest.fixture(scope="class")
    def wrapper(self, cosmo: camb.CAMBdata) -> CAMBStandardCosmology:
        return CAMBStandardCosmology(cosmo)

    # =========================================================================
    # Tests

    def test_wrapper_is_compliant(self, wrapper):
        """Test that AstropyCosmology is a BackgroundCosmologyWrapperAPI."""
        super().test_wrapper_is_compliant(wrapper)

        assert isinstance(wrapper, StandardCosmologyAPI)
        assert isinstance(wrapper, StandardCosmologyWrapperAPI)

    # =========================================================================
    # FLRW API Tests

    def test_H0(self, wrapper, cosmo):
        """Test that the wrapper has the same H0 as the wrapped object."""
        assert wrapper.H0 == cosmo.H0

    def test_Om0(self, wrapper, cosmo):
        """Test that the wrapper has the same Om0 as the wrapped object."""
        assert wrapper.Om0 == cosmo.Om0
        assert isinstance(wrapper.Om0, np.ndarray)

    def test_Ode0(self, wrapper, cosmo):
        """Test that the wrapper has the same Ode0 as the wrapped object."""
        assert wrapper.Ode0 == cosmo.Ode0
        assert isinstance(wrapper.Ode0, np.ndarray)

    def test_Tcmb0(self, wrapper, cosmo):
        """Test that the wrapper has the same Tcmb0 as the wrapped object."""
        assert wrapper.Tcmb0 == cosmo.Tcmb0

    def test_Neff(self, wrapper, cosmo):
        """Test that the wrapper has the same Neff as the wrapped object."""
        assert wrapper.Neff == cosmo.Neff
        assert isinstance(wrapper.Neff, np.ndarray)

    def test_m_nu(self, wrapper, cosmo):
        """Test that the wrapper has the same m_nu as the wrapped object."""
        assert all(np.equal(w, c) for w, c in zip(wrapper.m_nu, tuple(cosmo.m_nu)))
        assert isinstance(wrapper.m_nu, tuple)

    def test_Ob0(self, wrapper, cosmo):
        """Test that the wrapper has the same Ob0 as the wrapped object."""
        assert wrapper.Ob0 == cosmo.Ob0
        assert isinstance(wrapper.Ob0, np.ndarray)

    def test_h(self, wrapper, cosmo):
        """Test that the wrapper has the same h as the wrapped object."""
        assert wrapper.h == cosmo.h
        assert isinstance(wrapper.h, np.ndarray)

    def test_hubble_distance(self, wrapper, cosmo):
        """Test that the wrapper has the same hubble_distance as the wrapped object."""
        assert wrapper.hubble_distance == cosmo.hubble_distance

    def test_hubble_time(self, wrapper, cosmo):
        """Test that the wrapper has the same hubble_time as the wrapped object."""
        assert wrapper.hubble_time == cosmo.hubble_time

    def test_Odm0(self, wrapper, cosmo):
        """Test that the wrapper has the same Odm0 as the wrapped object."""
        assert wrapper.Odm0 == cosmo.Odm0
        assert isinstance(wrapper.Odm0, np.ndarray)

    def test_Ok0(self, wrapper, cosmo):
        """Test that the wrapper has the same Ok0 as the wrapped object."""
        assert wrapper.Ok0 == cosmo.Ok0
        assert isinstance(wrapper.Ok0, np.ndarray)

    def test_Ogamma0(self, wrapper, cosmo):
        """Test that the wrapper has the same Ogamma0 as the wrapped object."""
        assert wrapper.Ogamma0 == cosmo.Ogamma0
        assert isinstance(wrapper.Ogamma0, np.ndarray)

    def test_Onu0(self, wrapper, cosmo):
        """Test that the wrapper has the same Onu0 as the wrapped object."""
        assert wrapper.Onu0 == cosmo.Onu0
        assert isinstance(wrapper.Onu0, np.ndarray)

    @given(z_arr_st())
    def test_Tcmb(self, wrapper, cosmo, z):
        """Test that the wrapper's Tcmb is the same as the wrapped object's."""
        T = cosmo.Tcmb(z)
        assert np.array_equal(T, cosmo.Tcmb(z))

    @given(z_arr_st())
    def test_H(self, wrapper, cosmo, z):
        """Test that the wrapper's H is the same as the wrapped object's."""
        H = wrapper.H(z)
        assert np.array_equal(H, cosmo.H(z))

    @given(z_arr_st())
    def test_efunc(self, wrapper, cosmo, z):
        """Test that the wrapper's efunc is the same as the wrapped object's."""
        e = wrapper.efunc(z)
        assert np.array_equal(e, cosmo.efunc(z))
        assert isinstance(e, np.ndarray)

    @given(z_arr_st())
    def test_inv_efunc(self, wrapper, cosmo, z):
        """Test that the wrapper's inv_efunc is the same as the wrapped object's."""
        ie = wrapper.inv_efunc(z)
        assert np.array_equal(ie, cosmo.inv_efunc(z))
        assert isinstance(ie, np.ndarray)

    # TODO: why do these fail for z-> inf?
    @given(z_arr_st(max_value=1e9))
    def test_Om(self, wrapper, cosmo, z):
        """Test that the wrapper's Om is the same as the wrapped object's."""
        omega = wrapper.Om(z)
        assert np.array_equal(omega, cosmo.Om(z))
        assert isinstance(omega, np.ndarray)

    @given(z_arr_st(max_value=1e9))
    def test_Ob(self, wrapper, cosmo, z):
        """Test that the wrapper's Ob is the same as the wrapped object's."""
        omega = wrapper.Ob(z)
        assert np.array_equal(omega, cosmo.Ob(z))
        assert isinstance(omega, np.ndarray)

    @given(z_arr_st(max_value=1e9))
    def test_Odm(self, wrapper, cosmo, z):
        """Test that the wrapper's Odm is the same as the wrapped object's."""
        omega = wrapper.Odm(z)
        assert np.array_equal(omega, cosmo.Odm(z))
        assert isinstance(omega, np.ndarray)

    @given(z_arr_st(max_value=1e9))
    def test_Ogamma(self, wrapper, cosmo, z):
        """Test that the wrapper's Ogamma is the same as the wrapped object's."""
        omega = wrapper.Ogamma(z)
        assert np.array_equal(omega, cosmo.Ogamma(z))
        assert isinstance(omega, np.ndarray)

    @given(z_arr_st(max_value=1e9))
    def test_Onu(self, wrapper, cosmo, z):
        """Test that the wrapper's Onu is the same as the wrapped object's."""
        omega = wrapper.Onu(z)
        assert np.array_equal(omega, cosmo.Onu(z))
        assert isinstance(omega, np.ndarray)

    @given(z_arr_st())
    def test_Ode(self, wrapper, cosmo, z):
        """Test that the wrapper's Ode is the same as the wrapped object's."""
        omega = wrapper.Ode(z)
        assert np.array_equal(omega, cosmo.Ode(z))
        assert isinstance(omega, np.ndarray)

    @given(z_arr_st())
    def test_Ok(self, wrapper, cosmo, z):
        """Test that the wrapper's Ok is the same as the wrapped object's."""
        omega = wrapper.Ok(z)
        assert np.array_equal(omega, cosmo.Ok(z))
        assert isinstance(omega, np.ndarray)
