"""Test the Cosmology API compat library."""

from cosmology.api import CosmologyConstantsAPINamespace
from cosmology.compat.camb import constants


def test_namespace_is_compliant():
    """Test :mod:`cosmology.compat.camb.constants`."""
    assert isinstance(constants, CosmologyConstantsAPINamespace)
