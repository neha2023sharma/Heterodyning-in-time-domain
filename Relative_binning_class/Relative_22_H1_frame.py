#!/usr/bin/env python
# coding: utf-8
"""
Backward-compatibility shim — imports unified class from relative_binning.py.
"""
from relative_binning import (  # noqa: F401
    RelativeBinningLikelihood22,
    RelativeBinningTimeDomainGeocentTimeFrame,
    RelativeBinningTimeDomainH1detectorframe,
    Set_Fiducial_parameters,
)
