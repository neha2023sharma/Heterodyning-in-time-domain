#!/usr/bin/env python
# coding: utf-8
"""
Shared base class for all time-domain likelihood implementations.

Provides:
- Gohberg-Semencul C^{-1} v computation (Inner_product_C_inv_vector)
- weighted_inner_product
- injection_parameters / Data_list properties
- create_lalparams_with_modes (for exact-likelihood subclasses)
- noise_log_likelihood / log_likelihood
- _time_delay / _t_gps helpers (frame-dependent, used by subclasses)
"""
import bilby
import numpy as np
import lalsimulation as ls
import lal
from scipy.linalg import matmul_toeplitz


class TimeDomainLikelihoodBase(bilby.Likelihood):
    """
    Abstract base for time-domain GW likelihoods.

    Subclasses must set self._time_key and implement _time_delay().
    """

    # ------------------------------------------------------------------
    # Gohberg-Semencul C^{-1} v
    # ------------------------------------------------------------------

    def Inner_product_C_inv_vector(self, x, y, v):
        """
        Compute C^{-1} v using the Gohberg-Semencul representation.

        C^{-1} v = (1/x[0]) [ L(x) L(x)^T v - L(Jy) L(Jy)^T v ]

        where L(w) is the lower-triangular Toeplitz matrix with first column w.
        Each matrix-vector product is O(N log N) via FFT (matmul_toeplitz).
        """
        xf = np.concatenate(([x[0]], np.zeros(len(x) - 1)))
        ys = np.concatenate(([0.0], y[:-1]))
        zs = np.zeros(len(x))
        return (1.0 / x[0]) * (
            matmul_toeplitz((x, xf), matmul_toeplitz((xf, x), v))
            - matmul_toeplitz((ys, zs), matmul_toeplitz((zs, ys), v))
        )

    def weighted_inner_product(self, x, y, h1, h2):
        """Return (h1, C^{-1} h2)."""
        return np.dot(h1, self.Inner_product_C_inv_vector(x, y, h2))

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def injection_parameters(self):
        return self._injection_parameters

    @injection_parameters.setter
    def injection_parameters(self, new_params):
        self._injection_parameters = new_params.copy()
        self._Data_list = {}

    @property
    def Data_list(self):
        return self._Data_list

    # ------------------------------------------------------------------
    # LAL mode setup (used by exact-likelihood subclasses)
    # ------------------------------------------------------------------

    def create_lalparams_with_modes(self):
        lalParams = lal.CreateDict()
        ModeArray = ls.SimInspiralCreateModeArray()
        for l, m in self.modes_to_activate:
            ls.SimInspiralModeArrayActivateMode(ModeArray, l, m)
            ls.SimInspiralModeArrayActivateMode(ModeArray, l, -m)
        ls.SimInspiralWaveformParamsInsertModeArray(lalParams, ModeArray)
        self.lalParams = lalParams

    # ------------------------------------------------------------------
    # Frame-dependent time helpers (overridden/used by subclasses)
    # ------------------------------------------------------------------

    def _time_delay(self, params, detector):
        """
        Return the time delay to `detector` for the given parameter dict.
        Dispatches on self._time_key set by the subclass constructor.
        """
        if self._time_key == "geocent_time":
            return detector.time_delay_from_earth_center(
                right_ascension=params["ra"],
                declination=params["dec"],
                t_gps=params["geocent_time"],
            )
        else:
            return detector.time_delay_from_detector(
                other_detector=self.Detectors_list["H1"],
                right_ascension=params["ra"],
                declination=params["dec"],
                t_gps=params["H1_time"],
            )

    def _antenna_pattern(self, params, detector):
        """Return (F_plus, F_cross) for detector at the given parameters."""
        return detector.antenna_pattern(
            right_ascension=params["ra"],
            declination=params["dec"],
            polarization=params["psi"],
            t_gps=params[self._time_key],
        )

    # ------------------------------------------------------------------
    # Likelihoods
    # ------------------------------------------------------------------

    def noise_log_likelihood(self):
        if self.noise_log_likelihood_value is None:
            self.noise_log_likelihood_value = -0.5 * sum(
                self.data_times_C_inv_times_data[k]
                for k in self.Detectors_list
            )
        return self.noise_log_likelihood_value

    def log_likelihood(self, **parameters):
        return self.log_likelihood_ratio(**parameters) + self.noise_log_likelihood()
