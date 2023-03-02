"""Test the Cosmology API compat library."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given
from hypothesis.extra import numpy as npst

from cosmology.api import BackgroundCosmologyAPI, BackgroundCosmologyWrapperAPI
from cosmology.compat.camb import CAMBBackgroundCosmology

from .test_core import Test_CAMBCosmology

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


class Test_CAMBBackgroundCosmology(Test_CAMBCosmology):
    @pytest.fixture(scope="class")
    def wrapper(self, cosmo):
        return CAMBBackgroundCosmology(cosmo)

    # =========================================================================
    # Tests

    def test_wrapper_is_compliant(self, wrapper):
        """Test that AstropyCosmology is a BackgroundCosmologyWrapperAPI."""
        super().test_wrapper_is_compliant(wrapper)

        assert isinstance(wrapper, BackgroundCosmologyAPI)
        assert isinstance(wrapper, BackgroundCosmologyWrapperAPI)

    # =========================================================================
    # FLRW API Tests

    def test_scale_factor0(self, wrapper):
        """Test that scale_factor(0) returns 1."""
        assert wrapper.scale_factor0 == 1
        assert isinstance(wrapper.scale_factor0, np.ndarray)

    def test_Otot0(self, wrapper, cosmo):
        """Test that the wrapper has the same Otot0 as the wrapped object."""
        assert wrapper.Otot0 == cosmo.Otot0
        assert isinstance(wrapper.Otot0, np.ndarray)

    def test_critical_density0(self, wrapper, cosmo):
        """Test that the wrapper's critical_density0 is the same as critical_density0."""
        assert wrapper.critical_density0 == cosmo.critical_density0

    @given(z_arr_st(min_value=None))
    def test_scale_factor(self, wrapper, cosmo, z):
        """Test that the wrapper's scale_factor is the same as the wrapped object's."""
        a = cosmo.scale_factor(z)
        assert np.array_equal(a, cosmo.scale_factor(z))
        assert isinstance(a, np.ndarray)

    @given(z_arr_st())
    def test_Otot(self, wrapper, cosmo, z):
        """Test that the wrapper's Otot is the same as the wrapped object's."""
        omega = wrapper.Otot(z)
        assert np.array_equal(omega, cosmo.Otot(z))
        assert isinstance(omega, np.ndarray)

    @given(z_arr_st(max_value=1e9))
    def test_critical_density(self, wrapper, cosmo, z):
        r"""Test that the wrapper's critical_density is critical_density."""
        rho = wrapper.critical_density(z)
        assert np.array_equal(rho, cosmo.critical_density(z))

    @given(z_arr_st())
    def test_age(self, wrapper, cosmo, z):
        """Test the wrapper's age."""
        age = wrapper.age(z)
        assert np.array_equal(age, cosmo.age(z))

    @given(z_arr_st())
    def test_lookback_time(self, wrapper, cosmo, z):
        """Test the wrapper's lookback_time."""
        t = wrapper.lookback_time(z)
        assert np.array_equal(t, cosmo.lookback_time(z))

    @given(z_arr_st())
    def test_comoving_distance(self, wrapper, cosmo, z):
        """Test the wrapper's comoving_distance."""
        d = wrapper.comoving_distance(z)
        assert np.array_equal(d, cosmo.comoving_distance(z))

    @given(z_arr_st())
    def test_comoving_transverse_distance(self, wrapper, cosmo, z):
        """Test the wrapper's comoving_transverse_distance."""
        d = wrapper.comoving_transverse_distance(z)
        assert np.array_equal(d, cosmo.comoving_transverse_distance(z))

    @given(z_arr_st())
    def test_comoving_volume(self, wrapper, cosmo, z):
        """Test the wrapper's comoving_volume."""
        v = wrapper.comoving_volume(z)
        assert np.array_equal(v, cosmo.comoving_volume(z))

    @given(z_arr_st())
    def test_differential_comoving_volume(self, wrapper, cosmo, z):
        """Test the wrapper's differential_comoving_volume."""
        v = wrapper.differential_comoving_volume(z)
        assert np.array_equal(v, cosmo.differential_comoving_volume(z))

    @given(z_arr_st())
    def test_angular_diameter_distance(self, wrapper, cosmo, z):
        """Test the wrapper's angular_diameter_distance."""
        d = wrapper.angular_diameter_distance(z)
        assert np.array_equal(d, cosmo.angular_diameter_distance(z))

    @given(z_arr_st())
    def test_luminosity_distance(self, wrapper, cosmo, z):
        """Test the wrapper's luminosity_distance."""
        d = wrapper.luminosity_distance(z)
        assert np.array_equal(d, cosmo.luminosity_distance(z))
