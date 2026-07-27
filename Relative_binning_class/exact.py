#!/usr/bin/env python
# coding: utf-8
"""
Exact time-domain likelihood (no heterodyning).

A single class parameterised by `frame`:
  frame='geocent'  → time delays from geocentre, samples geocent_time
  frame='H1'       → time delays relative to H1 detector, samples H1_time
"""
import numpy as np
import lalsimulation as ls
import lal
from bilby.gw.conversion import chirp_mass_and_mass_ratio_to_component_masses

from base import TimeDomainLikelihoodBase


class ExactLikelihoodTimeDomain(TimeDomainLikelihoodBase):
    """
    Exact time-domain Gaussian likelihood using Gohberg-Semencul C^{-1}.

    Parameters
    ----------
    time : 1D array
        Time samples at which data are defined.
    Detectors_list : dict
        Mapping detector name -> pycbc.detector.Detector instance.
    injection_parameters : dict
        True signal parameters.
    Data_list : dict or None
        Pre-computed data per detector.  If None or empty the data are
        generated from the injection.
    x, y : dict
        Gohberg-Semencul vectors per detector (keyed by detector name).
    Noise : dict
        Noise realisations per detector.
    fmin, fref : float
    modes_to_activate : list of (l, m) tuples
    frame : {'geocent', 'H1'}
        Reference frame for the arrival time parameter.
    """

    def __init__(
        self,
        time,
        Detectors_list,
        injection_parameters,
        Data_list,
        x,
        y,
        Noise,
        fmin=10,
        fref=10,
        modes_to_activate=[(2, 2)],
        frame="geocent",
    ):
        super().__init__()
        if frame not in ("geocent", "H1"):
            raise ValueError(f"frame must be 'geocent' or 'H1', got {frame!r}")
        self.frame = frame
        self._time_key = "geocent_time" if frame == "geocent" else "H1_time"

        self.time = time
        self.Detectors_list = Detectors_list
        self.injection_parameters = injection_parameters.copy()
        self.x = x
        self.y = y
        self.Noise = Noise
        self._Data_list = Data_list if Data_list else {}
        self.modes_to_activate = modes_to_activate
        self.noise_log_likelihood_value = None
        self.fmin = fmin
        self.fref = fref

        self.create_lalparams_with_modes()
        self.compute_SNR_TD_and_waveform_data()

    # ------------------------------------------------------------------
    # Waveform
    # ------------------------------------------------------------------

    def waveform(self, parameters, Time_array, fmin=None, fref=None, dT=1 / 4096.0):
        """Return complex time-domain waveform (h+ - i hx) for given parameters."""
        fmin = fmin if fmin is not None else self.fmin
        fref = fref if fref is not None else self.fref

        chirp_mass = parameters["chirp_mass"]
        mass_ratio = parameters["mass_ratio"]
        chi_1 = parameters["chi_1"]
        chi_2 = parameters["chi_2"]
        luminosity_distance = parameters["luminosity_distance"]
        theta_jn = parameters["theta_jn"]
        phase = parameters["phase"]

        m1, m2 = chirp_mass_and_mass_ratio_to_component_masses(chirp_mass, mass_ratio)
        t_seq = lal.CreateREAL8Sequence(len(Time_array))
        t_seq.data = Time_array / ((m1 + m2) * lal.MTSUN_SI)
        h = ls.SimIMRPhenomTHM_neha_truncate(
            m1 * lal.MSUN_SI,
            m2 * lal.MSUN_SI,
            chi_1,
            chi_2,
            luminosity_distance * 1e6 * lal.PC_SI,
            theta_jn,
            dT,
            fmin,
            fref,
            phase,
            t_seq,
            self.lalParams,
        )
        return h[0].data.data - 1j * h[1].data.data

    # ------------------------------------------------------------------
    # SNR / data setup (called once at construction)
    # ------------------------------------------------------------------

    def compute_SNR_TD_and_waveform_data(self):
        data_times_C_inv = {}
        data_times_C_inv_times_data = {}
        antenna_patterns = {}
        h_injection = {}
        Time_array_per_detector = {}
        t_delay_injection_dict = {}

        for k, det in self.Detectors_list.items():
            t_delay = self._time_delay(self.injection_parameters, det)
            Time_array_per_detector[k] = self.time + t_delay
            t_delay_injection_dict[k] = t_delay

            h_injection[k] = self.waveform(
                self.injection_parameters, Time_array_per_detector[k] - t_delay
            )
            F_plus, F_cross = self._antenna_pattern(self.injection_parameters, det)
            antenna_patterns[k] = (F_plus, F_cross)

            if k not in self.Data_list or not len(self.Data_list.get(k, [])):
                self.Data_list[k] = (
                    F_plus * np.real(h_injection[k])
                    - F_cross * np.imag(h_injection[k])
                    + self.Noise[k]
                )

            data_times_C_inv[k] = self.Inner_product_C_inv_vector(
                self.x[k], self.y[k], self.Data_list[k]
            )
            data_times_C_inv_times_data[k] = np.dot(
                data_times_C_inv[k], self.Data_list[k]
            )

        self.Time_array_per_detector = Time_array_per_detector
        self.t_delay_injection_dict = t_delay_injection_dict
        self.data_times_C_inv = data_times_C_inv
        self.data_times_C_inv_times_data = data_times_C_inv_times_data
        self.h_injection = h_injection

        SNR_dict = {}
        SNR_arr = []
        for k, det in self.Detectors_list.items():
            F_plus, F_cross = antenna_patterns[k]
            signal = F_plus * np.real(self.h_injection[k]) + F_cross * (
                -np.imag(self.h_injection[k])
            )
            hh = self.weighted_inner_product(self.x[k], self.y[k], signal, signal)
            dh = self.weighted_inner_product(
                self.x[k], self.y[k], self.Data_list[k], signal
            )
            SNR = np.abs(dh) / np.sqrt(np.abs(hh))
            SNR_arr.append(SNR)
            SNR_dict[k] = SNR

        return (
            SNR_dict,
            np.sqrt(np.sum(np.array(SNR_arr) ** 2)),
            self.h_injection,
            self.Data_list,
            self.data_times_C_inv,
            self.data_times_C_inv_times_data,
        )

    # ------------------------------------------------------------------
    # Likelihood
    # ------------------------------------------------------------------

    def log_likelihood_ratio(self, **parameters):
        """Exact log-likelihood ratio ln L(d|θ) - ln L(d|noise)."""
        log_like = 0.0
        for k, det in self.Detectors_list.items():
            t_delay = self._time_delay(parameters, det)
            h = self.waveform(
                parameters,
                self.Time_array_per_detector[k]
                - (
                    t_delay
                    + parameters[self._time_key]
                    - self.injection_parameters[self._time_key]
                ),
            )
            F_plus, F_cross = self._antenna_pattern(parameters, det)
            signal = F_plus * np.real(h) + F_cross * (-np.imag(h))
            log_like += np.dot(self.data_times_C_inv[k], signal) - 0.5 * (
                self.weighted_inner_product(self.x[k], self.y[k], signal, signal)
            )
        return log_like


# ---------------------------------------------------------------------------
# Backward-compatible aliases (subclass so isinstance checks still pass)
# ---------------------------------------------------------------------------

class ExactLikelihoodTimeDomainGeocentTimeFrame(ExactLikelihoodTimeDomain):
    """Alias for ExactLikelihoodTimeDomain(frame='geocent')."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("frame", "geocent")
        super().__init__(*args, **kwargs)


class ExactLikelihoodTimeDomainH1detectorframe(ExactLikelihoodTimeDomain):
    """Alias for ExactLikelihoodTimeDomain(frame='H1')."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("frame", "H1")
        super().__init__(*args, **kwargs)
