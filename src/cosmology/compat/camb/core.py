"""The Cosmology API compatability wrapper for CAMB."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import camb
from numpy import floating
from numpy.typing import NDArray
from typing_extensions import TypeAlias

from cosmology.api import CosmologyAPINamespace, CosmologyWrapperAPI

__all__: list[str] = []

##############################################################################
# PARAMETERS

NDFloating: TypeAlias = NDArray[floating[Any]]


##############################################################################


@dataclass(frozen=True)
class CAMBCosmology(CosmologyWrapperAPI[NDFloating]):
    """The Cosmology API wrapper for :mod:`camb`."""

    cosmo: camb.CAMBdata
    name: str | None = None

    def __post_init__(self) -> None:
        """Run-time post-processing.

        Note that if this module is c-compiled (e.g. with :mod:`mypyc`) that
        the type of ``self.cosmo`` must be ``CAMBdata`` at object creation
        and cannot be later processed here.
        """
        if not isinstance(self.cosmo, (camb.CAMBdata, camb.CAMBParams)):
            msg = (
                "cosmo must be a CAMBdata or CAMBParams instance, "
                f"not {type(self.cosmo)}"
            )
            raise TypeError(msg)
        elif isinstance(self.cosmo, camb.CAMBParams):
            cosmo = camb.get_background(self.cosmo)
            params = self.cosmo

            # Need to fix the type of cosmo to CAMBdata.
            object.__setattr__(self, "cosmo", cosmo)
        else:
            params = self.cosmo.Params

        self._params: camb.CAMBparams
        object.__setattr__(self, "_params", params)

    def __cosmology_namespace__(
        self,
        /,
        *,
        api_version: str | None = None,
    ) -> CosmologyAPINamespace:
        """Returns an object that has all the cosmology API functions on it.

        Parameters
        ----------
        api_version: Optional[str]
            string representing the version of the cosmology API specification
            to be returned, in ``'YYYY.MM'`` form, for example, ``'2020.10'``.
            If ``None``, it return the namespace corresponding to latest version
            of the cosmology API specification.  If the given version is invalid
            or not implemented for the given module, an error is raised.
            Default: ``None``.

            .. note:: currently only `None` is supported.

        Returns
        -------
        `CosmologyAPINamespace`
            An object representing the CAMB cosmology API namespace.
        """
        import cosmology.compat.camb as namespace

        return cast("CosmologyAPINamespace", namespace)
