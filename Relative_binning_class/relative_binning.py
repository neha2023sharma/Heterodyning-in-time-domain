#!/usr/bin/env python
# coding: utf-8
"""
Relative-binning (heterodyning) time-domain likelihood — 22-mode only.

A single class parameterised by `frame`:
  frame='geocent'  → time delays from geocentre, samples geocent_time
  frame='H1'       → time delays relative to H1 detector, samples H1_time
"""
import numpy as np
import lalsimulation as ls
import lal
from bilby.gw.conversion import chirp_mass_and_mass_ratio_to_component_masses
from scipy.optimize import differential_evolution

from base import TimeDomainLikelihoodBase
from exact import ExactLikelihoodTimeDomain


class RelativeBinningLikelihood22(TimeDomainLikelihoodBase):
    """
    Time-domain relative-binning likelihood using the (2,2) mode only.

    Parameters
    ----------
    time : 1D array
    Detectors_list : dict
        Mapping detector name -> pycbc.detector.Detector
    fiducial_parameters : dict or None
        If None, sampled from `priors`.
    injection_parameters : dict
    Data_list : dict or None
    x, y : dict
        Gohberg-Semencul vectors per detector.
    Noise : dict
    fmin, fref : float
    epsilon_array_per_detector : dict
    spacing : array-like
    spacing_times : array-like
    time_split_per_detector : dict
    priors : dict or None
    frame : {'geocent', 'H1'}
    """

    def __init__(
        self,
        time,
        Detectors_list,
        fiducial_parameters,
        injection_parameters,
        Data_list,
        x,
        y,
        Noise,
        fmin=10,
        fref=10,
        epsilon_array_per_detector={
            "H1": np.array([0.5, 0.1]),
            "L1": np.array([0.5, 0.1]),
            "V1": np.array([0.5, 0.1]),
        },
        spacing=np.array([1, 10, 100]),
        spacing_times=np.array([0.1, 0.2]),
        time_split_per_detector={"H1": -0.02, "L1": -0.02, "V1": -0.1},
        priors=None,
        frame="geocent",
    ):
        super().__init__()
        if frame not in ("geocent", "H1"):
            raise ValueError(f"frame must be 'geocent' or 'H1', got {frame!r}")
        self.frame = frame
        self._time_key = "geocent_time" if frame == "geocent" else "H1_time"

        self.gamma = np.array([5 / 8, 3 / 8, 1 / 4, -3 / 8])
        self.chi = 1
        self.time = time
        self.epsilon_array_per_detector = epsilon_array_per_detector
        self.spacing = spacing
        self.spacing_times = spacing_times
        self.time_split_per_detector = time_split_per_detector
        self.Detectors_list = Detectors_list
        self.Noise = Noise
        if fiducial_parameters is None:
            fiducial_parameters = priors.sample()
        self.fiducial_parameters = fiducial_parameters.copy()
        self.injection_parameters = injection_parameters.copy()
        self.noise_log_likelihood_value = None
        self.x = x
        self.y = y
        self.fmin = fmin
        self.fref = fref
        self._Data_list = Data_list if Data_list else {}

        self.compute_SNR_TD_and_waveform_data()
        self.setup_bins_per_detector()
        self.Summary_data()

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
    # Waveforms
    # ------------------------------------------------------------------

    def waveform(self, parameters, Time_array, fmin=None, fref=None, dT=1 / 4096.0):
        """Full (h+ - i hx) waveform via IMRPhenomT."""
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
        h = ls.SimIMRPhenomT_neha_truncate(
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
            None,
        )
        return h[0].data.data - 1j * h[1].data.data

    def waveform_22_modes(self, parameters, Time_array, fmin=None, fref=None, dT=1 / 4096.0):
        """22 mode via IMRPhenomT."""
        fmin = fmin if fmin is not None else self.fmin
        fref = fref if fref is not None else self.fref
        chirp_mass = parameters["chirp_mass"]
        mass_ratio = parameters["mass_ratio"]
        chi_1 = parameters["chi_1"]
        chi_2 = parameters["chi_2"]
        luminosity_distance = parameters["luminosity_distance"]
        phase = parameters["phase"]
        m1, m2 = chirp_mass_and_mass_ratio_to_component_masses(chirp_mass, mass_ratio)
        t_seq = lal.CreateREAL8Sequence(len(Time_array))
        t_seq.data = Time_array / ((m1 + m2) * lal.MTSUN_SI)
        h = ls.SimIMRPhenomT_neha_just_modes(
            m1 * lal.MSUN_SI,
            m2 * lal.MSUN_SI,
            chi_1,
            chi_2,
            luminosity_distance * 1e6 * lal.PC_SI,
            dT,
            fmin,
            fref,
            phase,
            t_seq,
            None,
        )
        return h.mode.data.data

    def waveform_22_modes_fiducial(self, parameters, Time_array, fmin=None, fref=None, dT=1 / 4096.0):
        """22 mode without theta_jn (fiducial, no spin-weighted harmonics)."""
        return self.waveform_22_modes(parameters, Time_array, fmin=fmin, fref=fref, dT=dT)

    # ------------------------------------------------------------------
    # SNR / data setup
    # ------------------------------------------------------------------

    def compute_SNR_TD_and_waveform_data(self):
        data_times_C_inv = {}
        data_times_C_inv_times_data = {}
        antenna_patterns = {}
        h22_fiducial = {}
        h_injection = {}
        Time_array_per_detector = {}
        t_delay_injection_dict = {}

        for k, det in self.Detectors_list.items():
            t_delay_injection = self._time_delay(self.injection_parameters, det)
            t_delay_fiducial = self._time_delay(self.fiducial_parameters, det)
            Time_array_per_detector[k] = self.time + t_delay_injection
            t_delay_injection_dict[k] = t_delay_injection

            h22_fiducial[k] = self.waveform_22_modes_fiducial(
                self.fiducial_parameters, Time_array_per_detector[k] - t_delay_fiducial
            )
            h_injection[k] = self.waveform(
                self.injection_parameters, Time_array_per_detector[k] - t_delay_injection
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
            data_times_C_inv_times_data[k] = np.dot(data_times_C_inv[k], self.Data_list[k])

        self.Time_array_per_detector = Time_array_per_detector
        self.t_delay_injection_dict = t_delay_injection_dict
        self.data_times_C_inv = data_times_C_inv
        self.data_times_C_inv_times_data = data_times_C_inv_times_data
        self.h22_fiducial = h22_fiducial
        self.h_injection = h_injection

        SNR_dict = {}
        SNR_arr = []
        for k, det in self.Detectors_list.items():
            F_plus, F_cross = antenna_patterns[k]
            signal = F_plus * np.real(h_injection[k]) + F_cross * (-np.imag(h_injection[k]))
            hh = self.weighted_inner_product(self.x[k], self.y[k], signal, signal)
            dh = self.weighted_inner_product(self.x[k], self.y[k], self.Data_list[k], signal)
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
    # Binning
    # ------------------------------------------------------------------

    def setup_bins_inspiral(self, Time_array, time_split, epsilon_array):
        d_alpha = []
        for g in self.gamma:
            if g < 0:
                d_alpha.append(2 * np.pi * self.chi / (min(abs(Time_array))) ** g)
            else:
                d_alpha.append(2 * np.pi * self.chi / (max(abs(Time_array))) ** g)
        d_alpha = np.array(d_alpha)

        d_phi_f = np.sum(
            np.sign(self.gamma) * d_alpha * (abs(Time_array[:, None]) ** self.gamma),
            axis=1,
        )

        mask1 = Time_array <= time_split
        mask2 = Time_array > time_split
        d_phi_f_regions = [d_phi_f[mask1], d_phi_f[mask2]]
        idx_offsets = [0, np.sum(mask1)]

        bin_edge_indices = []
        for region_idx, (d_phi_f_reg, e) in enumerate(zip(d_phi_f_regions, epsilon_array)):
            number_of_bins = int(abs((d_phi_f_reg[-1] - d_phi_f_reg[0]) / e))
            region_bin_edges = []
            for i in range(number_of_bins + 1):
                bin_idx = np.where(
                    d_phi_f_reg - d_phi_f_reg[0]
                    >= (i / number_of_bins) * (d_phi_f_reg[-1] - d_phi_f_reg[0])
                )[0]
                if len(bin_idx) > 0:
                    region_bin_edges.append(bin_idx[-1])
                else:
                    region_bin_edges.append(0)
            region_bin_edges = np.array(region_bin_edges) + idx_offsets[region_idx]
            bin_edge_indices.extend(region_bin_edges)

        bin_edge_index = np.unique(np.array(bin_edge_indices))
        time_on_edge = Time_array[bin_edge_index]
        bin_width = time_on_edge[1:] - time_on_edge[:-1]
        bin_centre = (time_on_edge[1:] + time_on_edge[:-1]) / 2
        number_of_bins = len(bin_centre)
        return (number_of_bins, bin_edge_index, bin_centre, bin_width, time_on_edge)

    def setup_bins(self, Time_array_full, time_split, epsilon_array):
        idx_inspiral_end = np.where(Time_array_full < -0.0)[0][-1]
        Time_array_inspiral = Time_array_full[: idx_inspiral_end + 1]
        bin_edge_index_insp = self.setup_bins_inspiral(
            Time_array_inspiral, time_split, epsilon_array
        )[1]

        spacing_t_less_t1 = self.spacing[0]
        spacing_t_t1_to_t2 = self.spacing[1]
        spacing_t_greater_t2 = self.spacing[2]

        idx_merger_ringdown_start = idx_inspiral_end
        if Time_array_full[-1] > self.spacing_times[0]:
            idx_t_t1 = np.where(Time_array_full > self.spacing_times[0])[0][0]
        else:
            idx_t_t1 = int(len(Time_array_full))
        if Time_array_full[-1] > self.spacing_times[1]:
            idx_t_t2 = np.where(Time_array_full > self.spacing_times[1])[0][0]
        else:
            idx_t_t2 = int(len(Time_array_full))

        bins_t_less_t1 = np.arange(idx_merger_ringdown_start, idx_t_t1, spacing_t_less_t1)
        bins_t_t1_to_t2 = (
            np.arange(idx_t_t1, idx_t_t2, spacing_t_t1_to_t2)
            if Time_array_full[-1] > self.spacing_times[0]
            else np.array([], dtype=int)
        )
        bins_t_greater_t2 = (
            np.arange(idx_t_t2, len(Time_array_full), spacing_t_greater_t2)
            if Time_array_full[-1] > self.spacing_times[1]
            else np.array([], dtype=int)
        )

        bin_edge_index = np.concatenate(
            (
                bin_edge_index_insp,
                bins_t_less_t1[1:],
                bins_t_t1_to_t2[1:],
                bins_t_greater_t2[1:],
            ),
            dtype=int,
        )

        if bin_edge_index[-1] != len(Time_array_full) - 1:
            bin_edge_index = np.insert(
                bin_edge_index, len(bin_edge_index), len(Time_array_full) - 1
            )
        time_on_edge = Time_array_full[bin_edge_index]
        bin_width = time_on_edge[1:] - time_on_edge[:-1]
        bin_centre = (time_on_edge[1:] + time_on_edge[:-1]) / 2
        number_of_bins = len(bin_centre)
        return (time_on_edge, number_of_bins, bin_edge_index, bin_centre, bin_width)

    def setup_bins_per_detector(self):
        time_on_edge_per_detector = {}
        number_of_bins_per_detector = {}
        bin_edge_index_per_detector = {}
        bin_centre_per_detector = {}
        bin_width_per_detector = {}
        for k in self.Detectors_list.keys():
            time_on_edge, number_of_bins, bin_edge_index, bin_centre, bin_width = self.setup_bins(
                self.Time_array_per_detector["H1"],
                self.time_split_per_detector[k],
                self.epsilon_array_per_detector[k],
            )
            time_on_edge_per_detector[k] = time_on_edge.copy() + self.t_delay_injection_dict[k]
            number_of_bins_per_detector[k] = number_of_bins
            bin_edge_index_per_detector[k] = bin_edge_index.copy()
            bin_centre_per_detector[k] = bin_centre.copy() + self.t_delay_injection_dict[k]
            bin_width_per_detector[k] = bin_width.copy()

        self.time_on_edge = time_on_edge_per_detector
        self.number_of_bins = number_of_bins_per_detector
        self.bin_edge_index = bin_edge_index_per_detector
        self.bin_centre = bin_centre_per_detector
        self.bin_width = bin_width_per_detector

    # ------------------------------------------------------------------
    # Summary data
    # ------------------------------------------------------------------

    def Summary_data(self):
        Summary_data_dict = {}
        for k in self.Detectors_list.keys():
            time_diff_arr = np.zeros_like(self.Time_array_per_detector[k], dtype=float)
            for i in range(self.number_of_bins[k]):
                start_index = self.bin_edge_index[k][i]
                end_index = self.bin_edge_index[k][i + 1]
                time_diff_arr[start_index:end_index] = (
                    self.Time_array_per_detector[k][start_index:end_index]
                    - self.bin_centre[k][i]
                )

            data_times_C_inv_times_h22 = self.data_times_C_inv[k] * self.h22_fiducial[k]
            data_times_C_inv_times_h22_times_time_diff = (
                data_times_C_inv_times_h22 * time_diff_arr
            )

            A_0 = np.zeros(self.number_of_bins[k], dtype=complex)
            A_1 = np.zeros(self.number_of_bins[k], dtype=complex)
            H22_matrix = np.zeros(
                (len(self.h22_fiducial[k]), self.number_of_bins[k]), dtype=complex
            )
            H22_conj_matrix = np.zeros(
                (len(self.h22_fiducial[k]), self.number_of_bins[k]), dtype=complex
            )

            for i in range(self.number_of_bins[k]):
                start_index = self.bin_edge_index[k][i]
                end_index = (
                    self.bin_edge_index[k][i + 1] + 1
                    if i == self.number_of_bins[k] - 1
                    else self.bin_edge_index[k][i + 1]
                )
                A_0[i] = np.sum(data_times_C_inv_times_h22[start_index:end_index])
                A_1[i] = np.sum(
                    data_times_C_inv_times_h22_times_time_diff[start_index:end_index]
                )
                H22_matrix[:, i][start_index:end_index] = self.h22_fiducial[k][start_index:end_index]
                H22_conj_matrix[:, i][start_index:end_index] = np.conjugate(
                    self.h22_fiducial[k]
                )[start_index:end_index]

            B_0 = np.zeros((self.number_of_bins[k], self.number_of_bins[k]), dtype=complex)
            B_1 = np.zeros((self.number_of_bins[k], self.number_of_bins[k]), dtype=complex)
            B_2 = np.zeros((self.number_of_bins[k], self.number_of_bins[k]), dtype=complex)
            B_3 = np.zeros((self.number_of_bins[k], self.number_of_bins[k]), dtype=complex)

            h22_times_C_inv_full = self.Inner_product_C_inv_vector(
                self.x[k], self.y[k], H22_matrix
            )
            h22_conj_times_C_inv_full = self.Inner_product_C_inv_vector(
                self.x[k], self.y[k], H22_conj_matrix
            )

            for i in range(self.number_of_bins[k]):
                h22_ci = h22_times_C_inv_full[:, i]
                h22_conj_ci = h22_conj_times_C_inv_full[:, i]
                h22_ci_h22 = h22_ci * self.h22_fiducial[k]
                h22_conj_ci_h22 = h22_conj_ci * self.h22_fiducial[k]
                h22_ci_h22_td = h22_ci_h22 * time_diff_arr
                h22_conj_ci_h22_td = h22_conj_ci_h22 * time_diff_arr

                for j in range(self.number_of_bins[k]):
                    s = self.bin_edge_index[k][j]
                    e = (
                        self.bin_edge_index[k][j + 1] + 1
                        if j == self.number_of_bins[k] - 1
                        else self.bin_edge_index[k][j + 1]
                    )
                    B_0[i, j] = np.sum(h22_ci_h22[s:e])
                    B_1[i, j] = np.sum(h22_ci_h22_td[s:e])
                    B_2[i, j] = np.sum(h22_conj_ci_h22[s:e])
                    B_3[i, j] = np.sum(h22_conj_ci_h22_td[s:e])

            Summary_data_dict[k] = (A_0, A_1, B_0, B_1, B_2, B_3)
        self.Summary_data_dict = Summary_data_dict

    # ------------------------------------------------------------------
    # Spherical harmonics
    # ------------------------------------------------------------------

    @staticmethod
    def SphericalHarmonics22(theta, phi, m):
        return (
            np.sqrt(5.0 / (64.0 * np.pi))
            * (1.0 + np.cos(theta))
            * (1.0 + np.cos(theta))
            * np.exp(1j * m * phi)
        )

    @staticmethod
    def SphericalHarmonics2minus2(theta, phi, m):
        return (
            np.sqrt(5.0 / (64.0 * np.pi))
            * (1.0 - np.cos(theta))
            * (1.0 - np.cos(theta))
            * np.exp(1j * m * phi)
        )

    # ------------------------------------------------------------------
    # Likelihood
    # ------------------------------------------------------------------

    def log_likelihood_ratio(self, **parameters):
        log_like = 0
        Y_22 = self.SphericalHarmonics22(
            parameters["theta_jn"], np.pi / 2 - parameters["phase"], 2
        )
        Y_2_minus_2 = self.SphericalHarmonics2minus2(
            parameters["theta_jn"], np.pi / 2 - parameters["phase"], -2
        )
        for k, det in self.Detectors_list.items():
            A_0, A_1, B_0, B_1, B_2, B_3 = self.Summary_data_dict[k]

            h22 = self.waveform_22_modes(
                parameters,
                self.time[self.bin_edge_index[k]]
                - (parameters[self._time_key] - self.injection_parameters[self._time_key]),
            )

            F_plus, F_cross = self._antenna_pattern(parameters, det)

            waveform_ratio = h22 / self.h22_fiducial[k][self.bin_edge_index[k]]
            r_0 = (waveform_ratio[1:] + waveform_ratio[:-1]) / 2
            r_1 = (waveform_ratio[1:] - waveform_ratio[:-1]) / self.bin_width[k]

            Inner_product_d_h22 = np.dot(r_0, A_0) + np.dot(r_1, A_1)
            Y_22_Y_2_minus_2_d_h22 = (
                Y_22 * Inner_product_d_h22
                + Y_2_minus_2 * np.conjugate(Inner_product_d_h22)
            )
            Inner_product_di_sj = F_plus * np.real(
                Y_22_Y_2_minus_2_d_h22
            ) - F_cross * np.imag(Y_22_Y_2_minus_2_d_h22)

            B0_r0 = np.dot(B_0, r_0)
            B1_r1 = np.dot(B_1, r_1)
            B1_r0 = np.dot(B_1, r_0)
            Inner_product_h22_h22 = np.dot(r_0, (B0_r0 + B1_r1)) + np.dot(r_1, B1_r0)

            B2_r0 = np.dot(B_2, r_0)
            B3_r1 = np.dot(B_3, r_1)
            B3_r0 = np.dot(B_3, r_0)
            Inner_product_h22_conj_h22 = np.dot(
                np.conjugate(r_0), (B2_r0 + B3_r1)
            ) + np.dot(np.conjugate(r_1), B3_r0)

            Inner_product_sum_h22_sum_h22 = (
                Y_22 ** 2 * Inner_product_h22_h22
                + 2 * Y_22 * Y_2_minus_2 * Inner_product_h22_conj_h22
                + Y_2_minus_2 ** 2 * np.conjugate(Inner_product_h22_h22)
            )
            Inner_product_sum_h22_conj_sum_h22 = np.conjugate(Y_22) * (
                Y_22 * Inner_product_h22_conj_h22
                + Y_2_minus_2 * np.conjugate(Inner_product_h22_h22)
            ) + Y_2_minus_2 * (
                Y_22 * Inner_product_h22_h22
                + np.conjugate(Y_2_minus_2) * Inner_product_h22_conj_h22
            )
            Inner_product_si_sj = (
                0.5 * (F_plus ** 2 + F_cross ** 2) * np.real(Inner_product_sum_h22_conj_sum_h22)
                + 0.5 * (F_plus ** 2 - F_cross ** 2) * np.real(Inner_product_sum_h22_sum_h22)
                - F_plus * F_cross * np.imag(
                    Inner_product_sum_h22_sum_h22 + Inner_product_sum_h22_conj_sum_h22
                )
            )
            log_like += Inner_product_di_sj - 0.5 * Inner_product_si_sj
        return log_like


# ---------------------------------------------------------------------------
# Backward-compatible aliases
# ---------------------------------------------------------------------------

class RelativeBinningTimeDomainGeocentTimeFrame(RelativeBinningLikelihood22):
    """Alias for RelativeBinningLikelihood22(frame='geocent')."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("frame", "geocent")
        super().__init__(*args, **kwargs)


class RelativeBinningTimeDomainH1detectorframe(RelativeBinningLikelihood22):
    """Alias for RelativeBinningLikelihood22(frame='H1')."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("frame", "H1")
        super().__init__(*args, **kwargs)


# ---------------------------------------------------------------------------
# Fiducial parameter optimization
# ---------------------------------------------------------------------------

class Set_Fiducial_parameters(ExactLikelihoodTimeDomain):
    """
    Optimizes fiducial parameters for relative binning using the exact likelihood.
    """

    def __init__(
        self,
        time,
        Detectors_list,
        fiducial_parameters,
        injection_parameters,
        Data_list,
        x,
        y,
        Noise,
        priors,
        fmin=10,
        fref=10,
        parameters_to_be_updated=["chirp_mass", "mass_ratio"],
    ):
        super().__init__(
            time,
            Detectors_list,
            injection_parameters,
            Data_list,
            x,
            y,
            Noise,
            fmin,
            fref,
            frame="geocent",
        )
        self.fiducial_parameters = fiducial_parameters.copy()
        self.priors = priors.copy()
        self.parameters_to_be_updated = parameters_to_be_updated.copy()

    def lnlike_scipy_maximize(self, parameters):
        parameters = self.get_parameter_dictionary_from_list(parameters)
        return -self.log_likelihood_ratio(**parameters)

    def get_parameter_list_from_dictionary(self, parameter_dict):
        return [parameter_dict[k] for k in self.parameters_to_be_updated]

    def get_parameter_dictionary_from_list(self, parameter_list):
        return dict(zip(self.parameters_to_be_updated, parameter_list))

    def get_bounds_from_priors(self, priors_temp):
        return [
            [priors_temp[key].minimum, priors_temp[key].maximum]
            for key in self.parameters_to_be_updated
        ]

    def optimize_fiducial_parameters(self, iterations=5):
        parameters = self.fiducial_parameters.copy()
        old_ln_likelihood = self.log_likelihood_ratio(**parameters)
        priors_temp = self.priors.copy()
        for k in self.fiducial_parameters.keys():
            if k not in self.parameters_to_be_updated:
                priors_temp[k] = self.fiducial_parameters[k]

        params_bounds = self.get_bounds_from_priors(priors_temp)
        print(params_bounds)
        for it in range(iterations):
            x_0 = self.get_parameter_list_from_dictionary(parameters)
            result = differential_evolution(
                self.lnlike_scipy_maximize, bounds=params_bounds, x0=x_0
            )
            print(x_0)
            updated_params = self.get_parameter_dictionary_from_list(result.x)
            parameters.update(updated_params)
            new_ln_likelihood = self.log_likelihood_ratio(**parameters)
            print("New, old ln : ", new_ln_likelihood, old_ln_likelihood)
            if np.abs(new_ln_likelihood - old_ln_likelihood) < 0.1:
                break
            old_ln_likelihood = new_ln_likelihood

        return parameters
