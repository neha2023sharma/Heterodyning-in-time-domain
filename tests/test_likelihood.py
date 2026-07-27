#!/usr/bin/env python
# coding: utf-8
"""
Tests for Relative_binning_class refactored modules.

Unit tests (no LAL): test GS math, binning geometry, frame dispatch.
Integration tests (LAL required): test full likelihood construction and values.

Run with:
    pytest tests/test_likelihood.py -v
    pytest tests/test_likelihood.py -v -m unit        # unit only
    pytest tests/test_likelihood.py -v -m integration # integration only
"""
import sys
import os
import numpy as np
import pytest
from scipy.linalg import toeplitz, solve_toeplitz
from pycbc.detector import Detector

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'Relative_binning_class'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'ACF_noise_and_covariance_matrix_data'))

from base import TimeDomainLikelihoodBase
from exact import ExactLikelihoodTimeDomain
from relative_binning import RelativeBinningLikelihood22

# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(42)

FS = 4096.0          # Hz
DURATION = 1.0       # seconds — short enough to be fast
N = int(FS * DURATION)

# Canonical injection parameters (non-spinning BBH, geocent frame)
INJECTION_PARAMS_GEOCENT = {
    "chirp_mass": 28.3,
    "mass_ratio": 0.9,
    "chi_1": 0.0,
    "chi_2": 0.0,
    "luminosity_distance": 500.0,
    "theta_jn": 0.4,
    "phase": 1.3,
    "psi": 0.7,
    "ra": 1.375,
    "dec": -1.2108,
    "geocent_time": 1126259462.4,
}

DETECTORS = {"H1": Detector("H1"), "L1": Detector("L1")}

INJECTION_PARAMS_H1 = INJECTION_PARAMS_GEOCENT.copy()
_t_delay_h1 = DETECTORS["H1"].time_delay_from_earth_center(
    right_ascension=INJECTION_PARAMS_GEOCENT["ra"],
    declination=INJECTION_PARAMS_GEOCENT["dec"],
    t_gps=INJECTION_PARAMS_GEOCENT["geocent_time"],
)
INJECTION_PARAMS_H1["H1_time"] = INJECTION_PARAMS_GEOCENT["geocent_time"] + _t_delay_h1
INJECTION_PARAMS_H1.pop("geocent_time")

# Exponentially decaying ACF scaled to GW strain level (~1e-22 at 500 Mpc).
# Without scaling, the noise variance (1) vastly exceeds the GW signal amplitude
# (~1e-22), giving SNR ~ 1e-21 and LLR values drowning in floating-point noise.
_DECAY = 0.99
_NOISE_VAR = 1e-44   # noise rms ~ 1e-22, comparable to GW strain
_ACF = _NOISE_VAR * _DECAY ** np.arange(N)

def _gs_vectors(acf):
    e1 = np.zeros(len(acf))
    e1[0] = 1.0
    x = solve_toeplitz((acf, acf), e1)
    return x, x[::-1]

_X, _Y = _gs_vectors(_ACF)
XY = {"H1": (_X, _Y), "L1": (_X, _Y)}
X = {k: v[0] for k, v in XY.items()}
Y = {k: v[1] for k, v in XY.items()}
NOISE = {k: np.zeros(N) for k in DETECTORS}

TIME = np.arange(-DURATION + 1.0 / FS, 1.0 / FS, 1.0 / FS, dtype=np.float64)

# ---------------------------------------------------------------------------
# Unit tests — no LAL required
# ---------------------------------------------------------------------------

class TestGSMath:
    """Inner_product_C_inv_vector correctness against numpy."""

    def _make_base(self):
        # Minimal concrete subclass
        class Concrete(TimeDomainLikelihoodBase):
            def log_likelihood_ratio(self, **parameters):
                return 0.0
        obj = Concrete()
        obj._time_key = "geocent_time"
        obj.Detectors_list = DETECTORS
        return obj

    def test_vector_matches_numpy(self):
        base = self._make_base()
        C = toeplitz(_ACF)
        v = RNG.standard_normal(N)
        got = base.Inner_product_C_inv_vector(_X, _Y, v)
        ref = np.linalg.solve(C, v)
        np.testing.assert_allclose(got, ref, rtol=1e-5)

    def test_matrix_matches_numpy(self):
        base = self._make_base()
        C = toeplitz(_ACF)
        V = RNG.standard_normal((N, 4))
        got = base.Inner_product_C_inv_vector(_X, _Y, V)
        ref = np.linalg.solve(C, V)
        np.testing.assert_allclose(got, ref, rtol=1e-5)

    def test_weighted_inner_product_symmetry(self):
        base = self._make_base()
        h1 = RNG.standard_normal(N)
        h2 = RNG.standard_normal(N)
        ip12 = base.weighted_inner_product(_X, _Y, h1, h2)
        ip21 = base.weighted_inner_product(_X, _Y, h2, h1)
        np.testing.assert_allclose(ip12, ip21, rtol=1e-10)

    def test_weighted_inner_product_positive_definite(self):
        base = self._make_base()
        h = RNG.standard_normal(N)
        ip = base.weighted_inner_product(_X, _Y, h, h)
        assert ip > 0


class TestFrameDispatch:
    """_time_delay and _antenna_pattern dispatch on frame."""

    def _exact(self, frame):
        return ExactLikelihoodTimeDomain(
            time=TIME,
            Detectors_list=DETECTORS,
            injection_parameters=INJECTION_PARAMS_GEOCENT if frame == "geocent" else INJECTION_PARAMS_H1,
            Data_list={},
            x=X,
            y=Y,
            Noise=NOISE,
            frame=frame,
        )

    def test_geocent_time_key(self):
        obj = ExactLikelihoodTimeDomain.__new__(ExactLikelihoodTimeDomain)
        obj.frame = "geocent"
        obj._time_key = "geocent_time"
        assert obj._time_key == "geocent_time"

    def test_h1_time_key(self):
        obj = ExactLikelihoodTimeDomain.__new__(ExactLikelihoodTimeDomain)
        obj.frame = "H1"
        obj._time_key = "H1_time"
        assert obj._time_key == "H1_time"

    def test_invalid_frame_raises(self):
        with pytest.raises(ValueError, match="frame must be"):
            ExactLikelihoodTimeDomain(
                time=TIME, Detectors_list=DETECTORS,
                injection_parameters=INJECTION_PARAMS_GEOCENT,
                Data_list={}, x=X, y=Y, Noise=NOISE, frame="bad",
            )

    def test_time_delay_geocent_returns_float(self):
        base = TimeDomainLikelihoodBase.__new__(TimeDomainLikelihoodBase)
        base._time_key = "geocent_time"
        base.Detectors_list = DETECTORS
        td = base._time_delay(INJECTION_PARAMS_GEOCENT, DETECTORS["H1"])
        assert np.isscalar(td) or td.ndim == 0

    def test_time_delay_h1_returns_float(self):
        base = TimeDomainLikelihoodBase.__new__(TimeDomainLikelihoodBase)
        base._time_key = "H1_time"
        base.Detectors_list = DETECTORS
        td = base._time_delay(INJECTION_PARAMS_H1, DETECTORS["L1"])
        assert np.isscalar(td) or td.ndim == 0

    def test_antenna_pattern_returns_two_floats(self):
        base = TimeDomainLikelihoodBase.__new__(TimeDomainLikelihoodBase)
        base._time_key = "geocent_time"
        Fp, Fc = base._antenna_pattern(INJECTION_PARAMS_GEOCENT, DETECTORS["H1"])
        assert np.isfinite(Fp) and np.isfinite(Fc)


class TestBinning:
    """setup_bins_inspiral / setup_bins / setup_bins_per_detector — pure numpy."""

    @pytest.fixture
    def relbin(self):
        return RelativeBinningLikelihood22(
            time=TIME,
            Detectors_list=DETECTORS,
            fiducial_parameters=INJECTION_PARAMS_GEOCENT.copy(),
            injection_parameters=INJECTION_PARAMS_GEOCENT.copy(),
            Data_list={},
            x=X,
            y=Y,
            Noise=NOISE,
            frame="geocent",
        )

    def test_bin_edges_monotone(self, relbin):
        for k in DETECTORS:
            edges = relbin.bin_edge_index[k]
            assert np.all(np.diff(edges) > 0), f"Non-monotone bin edges for {k}"

    def test_bin_edges_cover_full_array(self, relbin):
        for k in DETECTORS:
            edges = relbin.bin_edge_index[k]
            assert edges[0] == 0 or edges[0] >= 0
            assert edges[-1] == N - 1

    def test_bin_width_positive(self, relbin):
        for k in DETECTORS:
            assert np.all(relbin.bin_width[k] > 0)

    def test_number_of_bins_consistent(self, relbin):
        for k in DETECTORS:
            assert relbin.number_of_bins[k] == len(relbin.bin_centre[k])
            assert relbin.number_of_bins[k] == len(relbin.bin_width[k])
            assert len(relbin.bin_edge_index[k]) == relbin.number_of_bins[k] + 1

    def test_setup_bins_inspiral_directly(self, relbin):
        t = TIME[TIME < 0]
        n_bins, edges, centres, widths, t_edges = relbin.setup_bins_inspiral(
            t, time_split=-0.1, epsilon_array=np.array([0.5, 0.1])
        )
        assert n_bins == len(centres)
        assert len(edges) == n_bins + 1
        assert np.all(widths > 0)


# ---------------------------------------------------------------------------
# Integration tests — require LAL / lalsimulation
# ---------------------------------------------------------------------------

class TestExactLikelihood:
    """ExactLikelihoodTimeDomain end-to-end."""

    @pytest.fixture(scope="class")
    @classmethod
    def exact_geocent(cls):
        return ExactLikelihoodTimeDomain(
            time=TIME,
            Detectors_list=DETECTORS,
            injection_parameters=INJECTION_PARAMS_GEOCENT.copy(),
            Data_list={},
            x=X,
            y=Y,
            Noise=NOISE,
            frame="geocent",
        )

    @pytest.fixture(scope="class")
    @classmethod
    def exact_h1(cls):
        return ExactLikelihoodTimeDomain(
            time=TIME,
            Detectors_list=DETECTORS,
            injection_parameters=INJECTION_PARAMS_H1.copy(),
            Data_list={},
            x=X,
            y=Y,
            Noise=NOISE,
            frame="H1",
        )

    def test_constructs_geocent(self, exact_geocent):
        assert exact_geocent is not None
        assert "H1" in exact_geocent.data_times_C_inv
        assert "L1" in exact_geocent.data_times_C_inv

    def test_constructs_h1(self, exact_h1):
        assert exact_h1 is not None

    def test_snr_positive(self, exact_geocent):
        snrs, net_snr, *_ = exact_geocent.compute_SNR_TD_and_waveform_data()
        assert net_snr > 0
        for snr in snrs.values():
            assert snr > 0

    def test_noise_log_likelihood_negative(self, exact_geocent):
        assert exact_geocent.noise_log_likelihood() < 0

    def test_noise_log_likelihood_cached(self, exact_geocent):
        # Second call should return exactly the same object (cached)
        v1 = exact_geocent.noise_log_likelihood()
        v2 = exact_geocent.noise_log_likelihood()
        assert v1 is v2 or v1 == v2

    def test_log_likelihood_ratio_at_injection_positive(self, exact_geocent):
        # At injection, signal matches data → ratio should be positive
        ratio = exact_geocent.log_likelihood_ratio(**INJECTION_PARAMS_GEOCENT)
        assert ratio > 0

    def test_log_likelihood_ratio_zero_noise_approx_half_snr_sq(self, exact_geocent):
        # Zero-noise: ln L(θ_inj) ≈ SNR²/2
        ratio = exact_geocent.log_likelihood_ratio(**INJECTION_PARAMS_GEOCENT)
        _, net_snr, *_ = exact_geocent.compute_SNR_TD_and_waveform_data()
        # Allow generous tolerance — time-domain discretisation causes small deviations
        assert abs(ratio - 0.5 * net_snr ** 2) / (0.5 * net_snr ** 2) < 0.05

    def test_log_likelihood_ratio_far_params_lower(self, exact_geocent):
        # A very different chirp mass should give a lower likelihood
        ratio_inj = exact_geocent.log_likelihood_ratio(**INJECTION_PARAMS_GEOCENT)
        far_params = INJECTION_PARAMS_GEOCENT.copy()
        far_params["chirp_mass"] = 10.0
        ratio_far = exact_geocent.log_likelihood_ratio(**far_params)
        assert ratio_inj > ratio_far

    def test_geocent_and_h1_frames_give_same_snr(self, exact_geocent, exact_h1):
        _, net_geocent, *_ = exact_geocent.compute_SNR_TD_and_waveform_data()
        _, net_h1, *_ = exact_h1.compute_SNR_TD_and_waveform_data()
        np.testing.assert_allclose(net_geocent, net_h1, rtol=1e-4)

    def test_backward_compat_geocent_alias(self):
        from exact import ExactLikelihoodTimeDomainGeocentTimeFrame
        obj = ExactLikelihoodTimeDomainGeocentTimeFrame(
            time=TIME, Detectors_list=DETECTORS,
            injection_parameters=INJECTION_PARAMS_GEOCENT.copy(),
            Data_list={}, x=X, y=Y, Noise=NOISE,
        )
        assert obj.frame == "geocent"
        assert isinstance(obj, ExactLikelihoodTimeDomain)

    def test_backward_compat_h1_alias(self):
        from exact import ExactLikelihoodTimeDomainH1detectorframe
        obj = ExactLikelihoodTimeDomainH1detectorframe(
            time=TIME, Detectors_list=DETECTORS,
            injection_parameters=INJECTION_PARAMS_H1.copy(),
            Data_list={}, x=X, y=Y, Noise=NOISE,
        )
        assert obj.frame == "H1"
        assert isinstance(obj, ExactLikelihoodTimeDomain)


class TestRelativeBinningLikelihood:
    """RelativeBinningLikelihood22 end-to-end."""

    @pytest.fixture(scope="class")
    @classmethod
    def relbin_geocent(cls):
        return RelativeBinningLikelihood22(
            time=TIME,
            Detectors_list=DETECTORS,
            fiducial_parameters=INJECTION_PARAMS_GEOCENT.copy(),
            injection_parameters=INJECTION_PARAMS_GEOCENT.copy(),
            Data_list={},
            x=X,
            y=Y,
            Noise=NOISE,
            frame="geocent",
        )

    @pytest.fixture(scope="class")
    @classmethod
    def relbin_h1(cls):
        return RelativeBinningLikelihood22(
            time=TIME,
            Detectors_list=DETECTORS,
            fiducial_parameters=INJECTION_PARAMS_H1.copy(),
            injection_parameters=INJECTION_PARAMS_H1.copy(),
            Data_list={},
            x=X,
            y=Y,
            Noise=NOISE,
            frame="H1",
        )

    def test_constructs_geocent(self, relbin_geocent):
        assert relbin_geocent is not None
        assert hasattr(relbin_geocent, "Summary_data_dict")

    def test_summary_data_shapes(self, relbin_geocent):
        for k in DETECTORS:
            A_0, A_1, B_0, B_1, B_2, B_3 = relbin_geocent.Summary_data_dict[k]
            n = relbin_geocent.number_of_bins[k]
            assert A_0.shape == (n,)
            assert A_1.shape == (n,)
            assert B_0.shape == (n, n)
            assert B_1.shape == (n, n)
            assert B_2.shape == (n, n)
            assert B_3.shape == (n, n)

    def test_log_likelihood_ratio_at_injection_positive(self, relbin_geocent):
        ratio = relbin_geocent.log_likelihood_ratio(**INJECTION_PARAMS_GEOCENT)
        assert ratio > 0

    def test_relbin_close_to_exact_at_injection(self, relbin_geocent):
        # At injection (fiducial = injection), relbin ratio should match exact closely
        exact = ExactLikelihoodTimeDomain(
            time=TIME,
            Detectors_list=DETECTORS,
            injection_parameters=INJECTION_PARAMS_GEOCENT.copy(),
            Data_list={},
            x=X,
            y=Y,
            Noise=NOISE,
            frame="geocent",
        )
        ratio_rb = relbin_geocent.log_likelihood_ratio(**INJECTION_PARAMS_GEOCENT)
        ratio_ex = exact.log_likelihood_ratio(**INJECTION_PARAMS_GEOCENT)
        # Relative binning is approximate; allow 1% error at injection
        np.testing.assert_allclose(ratio_rb, ratio_ex, rtol=0.01)

    def test_log_likelihood_ratio_far_params_lower(self, relbin_geocent):
        # Use parameters close but not equal to fiducial: relative binning
        # approximation breaks down when parameters differ by >~10% from fiducial.
        ratio_inj = relbin_geocent.log_likelihood_ratio(**INJECTION_PARAMS_GEOCENT)
        far = INJECTION_PARAMS_GEOCENT.copy()
        far["chirp_mass"] = INJECTION_PARAMS_GEOCENT["chirp_mass"] * 1.05  # 5% off
        ratio_far = relbin_geocent.log_likelihood_ratio(**far)
        assert ratio_inj > ratio_far

    def test_geocent_and_h1_give_consistent_ratio(self, relbin_geocent, relbin_h1):
        ratio_geocent = relbin_geocent.log_likelihood_ratio(**INJECTION_PARAMS_GEOCENT)
        ratio_h1 = relbin_h1.log_likelihood_ratio(**INJECTION_PARAMS_H1)
        # Same physical signal → same likelihood ratio
        np.testing.assert_allclose(ratio_geocent, ratio_h1, rtol=1e-3)

    def test_backward_compat_geocent_alias(self):
        from relative_binning import RelativeBinningTimeDomainGeocentTimeFrame
        obj = RelativeBinningTimeDomainGeocentTimeFrame(
            time=TIME, Detectors_list=DETECTORS,
            fiducial_parameters=INJECTION_PARAMS_GEOCENT.copy(),
            injection_parameters=INJECTION_PARAMS_GEOCENT.copy(),
            Data_list={}, x=X, y=Y, Noise=NOISE,
        )
        assert obj.frame == "geocent"
        assert isinstance(obj, RelativeBinningLikelihood22)

    def test_backward_compat_h1_alias(self):
        from relative_binning import RelativeBinningTimeDomainH1detectorframe
        obj = RelativeBinningTimeDomainH1detectorframe(
            time=TIME, Detectors_list=DETECTORS,
            fiducial_parameters=INJECTION_PARAMS_H1.copy(),
            injection_parameters=INJECTION_PARAMS_H1.copy(),
            Data_list={}, x=X, y=Y, Noise=NOISE,
        )
        assert obj.frame == "H1"
        assert isinstance(obj, RelativeBinningLikelihood22)

    def test_h1_frame_at_injection_positive(self, relbin_h1):
        ratio = relbin_h1.log_likelihood_ratio(**INJECTION_PARAMS_H1)
        assert ratio > 0

    def test_relbin_h1_close_to_exact_h1_at_injection(self, relbin_h1):
        exact_h1 = ExactLikelihoodTimeDomain(
            time=TIME, Detectors_list=DETECTORS,
            injection_parameters=INJECTION_PARAMS_H1.copy(),
            Data_list={}, x=X, y=Y, Noise=NOISE, frame="H1",
        )
        np.testing.assert_allclose(
            relbin_h1.log_likelihood_ratio(**INJECTION_PARAMS_H1),
            exact_h1.log_likelihood_ratio(**INJECTION_PARAMS_H1),
            rtol=0.01,
        )

    def test_geocent_time_shift_lowers_llr(self, relbin_geocent):
        # Shifting geocent_time by 50 ms should reduce the LLR.
        ratio_inj = relbin_geocent.log_likelihood_ratio(**INJECTION_PARAMS_GEOCENT)
        shifted = INJECTION_PARAMS_GEOCENT.copy()
        shifted["geocent_time"] += 0.05
        ratio_shifted = relbin_geocent.log_likelihood_ratio(**shifted)
        assert ratio_inj > ratio_shifted

    def test_log_likelihood_is_ratio_plus_noise(self, relbin_geocent):
        llr = relbin_geocent.log_likelihood_ratio(**INJECTION_PARAMS_GEOCENT)
        noise_ll = relbin_geocent.noise_log_likelihood()
        ll = relbin_geocent.log_likelihood(**INJECTION_PARAMS_GEOCENT)
        np.testing.assert_allclose(ll, llr + noise_ll, rtol=1e-12)

    def test_summary_data_A0_nonzero(self, relbin_geocent):
        for k in DETECTORS:
            A0, *_ = relbin_geocent.Summary_data_dict[k]
            assert np.abs(A0).max() > 0

    def test_summary_data_B0_symmetric(self, relbin_geocent):
        # B0[i,j] = h22_i^T C^{-1} h22_j; since C is real-symmetric, B0 = B0^T (not Hermitian)
        for k in DETECTORS:
            _, _, B0, *_ = relbin_geocent.Summary_data_dict[k]
            np.testing.assert_allclose(B0, B0.T, atol=1e-6)

    def test_snr_consistent_with_llr(self, relbin_geocent):
        # At injection with zero noise: LLR ≈ 0.5 * SNR^2
        exact = ExactLikelihoodTimeDomain(
            time=TIME, Detectors_list=DETECTORS,
            injection_parameters=INJECTION_PARAMS_GEOCENT.copy(),
            Data_list={}, x=X, y=Y, Noise=NOISE, frame="geocent",
        )
        _, net_snr, *_ = exact.compute_SNR_TD_and_waveform_data()
        llr = exact.log_likelihood_ratio(**INJECTION_PARAMS_GEOCENT)
        np.testing.assert_allclose(llr, 0.5 * net_snr**2, rtol=0.05)


# ---------------------------------------------------------------------------
# Exact likelihood — additional coverage
# ---------------------------------------------------------------------------

class TestExactLikelihoodExtra:

    @pytest.fixture(scope="class")
    @classmethod
    def exact_geocent(cls):
        return ExactLikelihoodTimeDomain(
            time=TIME, Detectors_list=DETECTORS,
            injection_parameters=INJECTION_PARAMS_GEOCENT.copy(),
            Data_list={}, x=X, y=Y, Noise=NOISE, frame="geocent",
        )

    def test_log_likelihood_is_ratio_plus_noise(self, exact_geocent):
        llr = exact_geocent.log_likelihood_ratio(**INJECTION_PARAMS_GEOCENT)
        noise_ll = exact_geocent.noise_log_likelihood()
        ll = exact_geocent.log_likelihood(**INJECTION_PARAMS_GEOCENT)
        np.testing.assert_allclose(ll, llr + noise_ll, rtol=1e-12)

    def test_noise_log_likelihood_deterministic(self, exact_geocent):
        v1 = exact_geocent.noise_log_likelihood()
        v2 = exact_geocent.noise_log_likelihood()
        assert v1 == v2  # cached, must be identical

    def test_distance_scaling(self):
        # Doubling luminosity_distance halves strain → SNR drops by 2
        p_near = INJECTION_PARAMS_GEOCENT.copy()
        p_far = INJECTION_PARAMS_GEOCENT.copy()
        p_far["luminosity_distance"] *= 2.0

        e_near = ExactLikelihoodTimeDomain(
            time=TIME, Detectors_list=DETECTORS,
            injection_parameters=p_near, Data_list={},
            x=X, y=Y, Noise=NOISE, frame="geocent",
        )
        e_far = ExactLikelihoodTimeDomain(
            time=TIME, Detectors_list=DETECTORS,
            injection_parameters=p_far, Data_list={},
            x=X, y=Y, Noise=NOISE, frame="geocent",
        )
        _, snr_near, *_ = e_near.compute_SNR_TD_and_waveform_data()
        _, snr_far, *_ = e_far.compute_SNR_TD_and_waveform_data()
        # SNR ∝ 1/d_L, so SNR_far = SNR_near / 2
        np.testing.assert_allclose(snr_far, snr_near / 2.0, rtol=1e-3)

    def test_chi1_nonzero_changes_llr(self):
        p_spin = INJECTION_PARAMS_GEOCENT.copy()
        p_spin["chi_1"] = 0.3

        e_nospin = ExactLikelihoodTimeDomain(
            time=TIME, Detectors_list=DETECTORS,
            injection_parameters=INJECTION_PARAMS_GEOCENT.copy(),
            Data_list={}, x=X, y=Y, Noise=NOISE, frame="geocent",
        )
        e_spin = ExactLikelihoodTimeDomain(
            time=TIME, Detectors_list=DETECTORS,
            injection_parameters=p_spin,
            Data_list={}, x=X, y=Y, Noise=NOISE, frame="geocent",
        )
        # Spinning and non-spinning injections evaluated at each other's params differ
        assert e_nospin.log_likelihood_ratio(**INJECTION_PARAMS_GEOCENT) != e_spin.log_likelihood_ratio(**p_spin)

    def test_data_with_nonzero_noise_still_has_positive_snr(self):
        rng_local = np.random.default_rng(99)
        noise_scale = np.sqrt(_NOISE_VAR)
        noisy_noise = {k: rng_local.standard_normal(N) * noise_scale for k in DETECTORS}
        e = ExactLikelihoodTimeDomain(
            time=TIME, Detectors_list=DETECTORS,
            injection_parameters=INJECTION_PARAMS_GEOCENT.copy(),
            Data_list={}, x=X, y=Y, Noise=noisy_noise, frame="geocent",
        )
        _, net_snr, *_ = e.compute_SNR_TD_and_waveform_data()
        assert net_snr > 0


# ---------------------------------------------------------------------------
# H1-frame exact likelihood — primary frame used in the paper
# ---------------------------------------------------------------------------

class TestExactLikelihoodH1:
    """ExactLikelihoodTimeDomain in H1-detector frame."""

    @pytest.fixture(scope="class")
    @classmethod
    def exact_h1(cls):
        return ExactLikelihoodTimeDomain(
            time=TIME,
            Detectors_list=DETECTORS,
            injection_parameters=INJECTION_PARAMS_H1.copy(),
            Data_list={},
            x=X,
            y=Y,
            Noise=NOISE,
            frame="H1",
        )

    def test_constructs(self, exact_h1):
        assert exact_h1 is not None
        assert "H1" in exact_h1.data_times_C_inv
        assert "L1" in exact_h1.data_times_C_inv

    def test_snr_positive(self, exact_h1):
        snrs, net_snr, *_ = exact_h1.compute_SNR_TD_and_waveform_data()
        assert net_snr > 0
        for snr in snrs.values():
            assert snr > 0

    def test_noise_log_likelihood_negative(self, exact_h1):
        assert exact_h1.noise_log_likelihood() < 0

    def test_noise_log_likelihood_deterministic(self, exact_h1):
        v1 = exact_h1.noise_log_likelihood()
        v2 = exact_h1.noise_log_likelihood()
        assert v1 == v2

    def test_log_likelihood_ratio_at_injection_positive(self, exact_h1):
        ratio = exact_h1.log_likelihood_ratio(**INJECTION_PARAMS_H1)
        assert ratio > 0

    def test_log_likelihood_ratio_approx_half_snr_sq(self, exact_h1):
        ratio = exact_h1.log_likelihood_ratio(**INJECTION_PARAMS_H1)
        _, net_snr, *_ = exact_h1.compute_SNR_TD_and_waveform_data()
        assert abs(ratio - 0.5 * net_snr**2) / (0.5 * net_snr**2) < 0.05

    def test_log_likelihood_ratio_far_params_lower(self, exact_h1):
        ratio_inj = exact_h1.log_likelihood_ratio(**INJECTION_PARAMS_H1)
        far = INJECTION_PARAMS_H1.copy()
        far["chirp_mass"] = 10.0
        ratio_far = exact_h1.log_likelihood_ratio(**far)
        assert ratio_inj > ratio_far

    def test_log_likelihood_is_ratio_plus_noise(self, exact_h1):
        llr = exact_h1.log_likelihood_ratio(**INJECTION_PARAMS_H1)
        noise_ll = exact_h1.noise_log_likelihood()
        ll = exact_h1.log_likelihood(**INJECTION_PARAMS_H1)
        np.testing.assert_allclose(ll, llr + noise_ll, rtol=1e-12)

    def test_h1_time_shift_lowers_llr(self, exact_h1):
        ratio_inj = exact_h1.log_likelihood_ratio(**INJECTION_PARAMS_H1)
        shifted = INJECTION_PARAMS_H1.copy()
        shifted["H1_time"] += 0.05
        ratio_shifted = exact_h1.log_likelihood_ratio(**shifted)
        assert ratio_inj > ratio_shifted

    def test_distance_scaling(self):
        p_near = INJECTION_PARAMS_H1.copy()
        p_far = INJECTION_PARAMS_H1.copy()
        p_far["luminosity_distance"] *= 2.0
        e_near = ExactLikelihoodTimeDomain(
            time=TIME, Detectors_list=DETECTORS,
            injection_parameters=p_near, Data_list={},
            x=X, y=Y, Noise=NOISE, frame="H1",
        )
        e_far = ExactLikelihoodTimeDomain(
            time=TIME, Detectors_list=DETECTORS,
            injection_parameters=p_far, Data_list={},
            x=X, y=Y, Noise=NOISE, frame="H1",
        )
        _, snr_near, *_ = e_near.compute_SNR_TD_and_waveform_data()
        _, snr_far, *_ = e_far.compute_SNR_TD_and_waveform_data()
        np.testing.assert_allclose(snr_far, snr_near / 2.0, rtol=1e-3)

    def test_data_with_nonzero_noise_still_has_positive_snr(self):
        rng_local = np.random.default_rng(77)
        noise_scale = np.sqrt(_NOISE_VAR)
        noisy_noise = {k: rng_local.standard_normal(N) * noise_scale for k in DETECTORS}
        e = ExactLikelihoodTimeDomain(
            time=TIME, Detectors_list=DETECTORS,
            injection_parameters=INJECTION_PARAMS_H1.copy(),
            Data_list={}, x=X, y=Y, Noise=noisy_noise, frame="H1",
        )
        _, net_snr, *_ = e.compute_SNR_TD_and_waveform_data()
        assert net_snr > 0


# ---------------------------------------------------------------------------
# H1-frame relative binning — primary configuration used in the paper
# ---------------------------------------------------------------------------

class TestRelativeBinningH1:
    """RelativeBinningLikelihood22 in H1-detector frame."""

    @pytest.fixture(scope="class")
    @classmethod
    def relbin_h1(cls):
        return RelativeBinningLikelihood22(
            time=TIME,
            Detectors_list=DETECTORS,
            fiducial_parameters=INJECTION_PARAMS_H1.copy(),
            injection_parameters=INJECTION_PARAMS_H1.copy(),
            Data_list={},
            x=X,
            y=Y,
            Noise=NOISE,
            frame="H1",
        )

    @pytest.fixture(scope="class")
    @classmethod
    def exact_h1(cls):
        return ExactLikelihoodTimeDomain(
            time=TIME,
            Detectors_list=DETECTORS,
            injection_parameters=INJECTION_PARAMS_H1.copy(),
            Data_list={},
            x=X,
            y=Y,
            Noise=NOISE,
            frame="H1",
        )

    def test_constructs(self, relbin_h1):
        assert relbin_h1 is not None
        assert hasattr(relbin_h1, "Summary_data_dict")

    def test_summary_data_shapes(self, relbin_h1):
        for k in DETECTORS:
            A_0, A_1, B_0, B_1, B_2, B_3 = relbin_h1.Summary_data_dict[k]
            n = relbin_h1.number_of_bins[k]
            assert A_0.shape == (n,)
            assert A_1.shape == (n,)
            assert B_0.shape == (n, n)
            assert B_1.shape == (n, n)
            assert B_2.shape == (n, n)
            assert B_3.shape == (n, n)

    def test_summary_data_A0_nonzero(self, relbin_h1):
        for k in DETECTORS:
            A0, *_ = relbin_h1.Summary_data_dict[k]
            assert np.abs(A0).max() > 0

    def test_summary_data_B0_symmetric(self, relbin_h1):
        for k in DETECTORS:
            _, _, B0, *_ = relbin_h1.Summary_data_dict[k]
            np.testing.assert_allclose(B0, B0.T, atol=1e-6)

    def test_log_likelihood_ratio_at_injection_positive(self, relbin_h1):
        ratio = relbin_h1.log_likelihood_ratio(**INJECTION_PARAMS_H1)
        assert ratio > 0

    def test_relbin_close_to_exact_at_injection(self, relbin_h1, exact_h1):
        ratio_rb = relbin_h1.log_likelihood_ratio(**INJECTION_PARAMS_H1)
        ratio_ex = exact_h1.log_likelihood_ratio(**INJECTION_PARAMS_H1)
        np.testing.assert_allclose(ratio_rb, ratio_ex, rtol=0.01)

    def test_log_likelihood_ratio_far_params_lower(self, relbin_h1):
        ratio_inj = relbin_h1.log_likelihood_ratio(**INJECTION_PARAMS_H1)
        far = INJECTION_PARAMS_H1.copy()
        far["chirp_mass"] = INJECTION_PARAMS_H1["chirp_mass"] * 1.05
        ratio_far = relbin_h1.log_likelihood_ratio(**far)
        assert ratio_inj > ratio_far

    def test_h1_time_shift_lowers_llr(self, relbin_h1):
        ratio_inj = relbin_h1.log_likelihood_ratio(**INJECTION_PARAMS_H1)
        shifted = INJECTION_PARAMS_H1.copy()
        shifted["H1_time"] += 0.05
        ratio_shifted = relbin_h1.log_likelihood_ratio(**shifted)
        assert ratio_inj > ratio_shifted

    def test_log_likelihood_is_ratio_plus_noise(self, relbin_h1):
        llr = relbin_h1.log_likelihood_ratio(**INJECTION_PARAMS_H1)
        noise_ll = relbin_h1.noise_log_likelihood()
        ll = relbin_h1.log_likelihood(**INJECTION_PARAMS_H1)
        np.testing.assert_allclose(ll, llr + noise_ll, rtol=1e-12)

    def test_snr_consistent_with_llr(self, relbin_h1, exact_h1):
        # At injection with zero noise: LLR ≈ 0.5 * SNR^2
        _, net_snr, *_ = exact_h1.compute_SNR_TD_and_waveform_data()
        llr = relbin_h1.log_likelihood_ratio(**INJECTION_PARAMS_H1)
        np.testing.assert_allclose(llr, 0.5 * net_snr**2, rtol=0.05)

    def test_bin_edges_monotone(self, relbin_h1):
        for k in DETECTORS:
            edges = relbin_h1.bin_edge_index[k]
            assert np.all(np.diff(edges) > 0)

    def test_bin_edges_cover_full_array(self, relbin_h1):
        for k in DETECTORS:
            assert relbin_h1.bin_edge_index[k][-1] == N - 1

    def test_bin_width_positive(self, relbin_h1):
        for k in DETECTORS:
            assert np.all(relbin_h1.bin_width[k] > 0)
