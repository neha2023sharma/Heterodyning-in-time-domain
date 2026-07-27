#!/usr/bin/env python
# coding: utf-8
"""
Backward-compatibility shim — imports unified class from exact.py.
"""
from exact import (  # noqa: F401
    ExactLikelihoodTimeDomain,
    ExactLikelihoodTimeDomainGeocentTimeFrame,
    ExactLikelihoodTimeDomainH1detectorframe,
)
