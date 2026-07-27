# Heterodyning in the Time Domain

Time-domain gravitational-wave likelihood using **relative binning (heterodyning)**, implemented for aligned-spin binary black holes with the IMRPhenomT waveform family.

This code accompanies the paper:

> **Rapid inference of gravitational-wave signals in the time domain using a heterodyned likelihood**  
> Neha Sharma, Aditya Vijaykumar, Prayush Kumar  
> [arXiv:2601.11239](https://arxiv.org/abs/2601.11239)

---

## Overview

Standard time-domain matched filtering requires evaluating the likelihood as an inner product `<d | h>` of strain data `d` against a template `h` via `C⁻¹`, the inverse noise covariance. For stationary Gaussian noise this is a symmetric Toeplitz matrix, inverted efficiently in O(N log N) via the **Gohberg–Semencul (GS) formula**.

**Relative binning** (heterodyning) reduces the cost of repeated likelihood evaluations during parameter estimation. Rather than re-evaluating `h` at every time sample, the waveform ratio `r(t) = h_test(t) / h_fiducial(t)` is approximated as piecewise linear within frequency-adaptive time bins. Pre-computed **summary data** (A₀, A₁, B₀–B₃) absorb the expensive C⁻¹ operations at construction time, leaving each likelihood call as a cheap dot product over O(N_bins) ≪ O(N) terms.

The binning strategy, including bin placement criteria, is described in [arXiv:1806.08792](https://arxiv.org/abs/1806.08792) and [arXiv:2601.11239](https://arxiv.org/abs/2601.11239).

---

## Repository layout

```
Relative_binning_class/
    base.py                  # Shared base class (GS solver, binning, SNR)
    exact.py                 # Exact time-domain likelihood
    relative_binning.py      # Relative-binning likelihood (22 mode)
    Exact_H1_frame.py        # Backward-compatible alias
    Exact_geocent_time.py    # Backward-compatible alias
    Relative_22_H1_frame.py  # Backward-compatible alias
    Relative_22_geocent_time.py

ACF_noise_and_covariance_matrix_data/
    Compute_x_and_y.py       # Generates GS vectors (x, y) from a PSD
    x_and_y_2_sec.pkl        # Pre-computed GS vectors (aLIGO O4 / AdV, 2 s)
    Noise.py

Examples/
    PE_relative_binning_22_mode_H1_time.py   # Full PE run (relbin)
    Exact_PE.py                              # Full PE run (exact)

Notebooks/
    Comparison_between_relbin_and_exact_log_likelihood.ipynb
    Geocent_to_H1_time_conversion.ipynb
    Time_dependent_antenna_patterns.ipynb
    Comparison_between_various_waveform_models.ipynb

scripts/
    build_lal_wheels.sh      # Build custom lalsuite fork + wire into uv env

tests/
    test_math.py             # Unit tests (no LAL): GS math, binning geometry
    test_likelihood.py       # Integration tests: full likelihood end-to-end
    conftest.py              # Sets DYLD_LIBRARY_PATH for lal shared libs
```

---

## Key classes

### `RelativeBinningLikelihood22` (`relative_binning.py`)

The main likelihood class. Parameterised by `frame`:

| `frame` | Time parameter sampled | Use case |
|---|---|---|
| `'H1'` | `H1_time` (arrival at LIGO-Hanford) | Paper results, recommended |
| `'geocent'` | `geocent_time` (geocentre) | General use |

```python
from relative_binning import RelativeBinningLikelihood22

likelihood = RelativeBinningLikelihood22(
    time=Time_array,
    Detectors_list={"H1": Detector("H1"), "L1": Detector("L1"), "V1": Detector("V1")},
    fiducial_parameters=fiducial_params,
    injection_parameters=injection_params,
    Data_list={},          # populated automatically at injection if empty
    x=x, y=y,             # GS vectors from Compute_x_and_y.py
    Noise={"H1": 0, "L1": 0, "V1": 0},
    fmin=10, fref=20,
    frame="H1",
)
snr_dict, network_snr, *_ = likelihood.compute_SNR_TD_and_waveform_data()
```

Backward-compatible class names are preserved:
- `RelativeBinningTimeDomainH1detectorframe` → `frame='H1'`
- `RelativeBinningTimeDomainGeocentTimeFrame` → `frame='geocent'`

### `ExactLikelihoodTimeDomain` (`exact.py`)

Exact (non-approximated) time-domain likelihood. Same interface, no `fiducial_parameters`. Used to validate the relative binning approximation.

---

## Installation

### Dependencies

- Python ≥ 3.10
- [bilby](https://github.com/bilby-dev/bilby) ≥ 2.3
- [pycbc](https://pycbc.org) ≥ 2.3
- A custom fork of **lalsuite** with the `IMRPhenomT_neha` waveform functions (not on PyPI — see below)

### 1. Build the custom lalsuite fork

```bash
bash scripts/build_lal_wheels.sh
```

This clones the fork, builds `lal` and `lalsimulation` via autotools, and wires them into the `uv` virtual environment via a `.pth` file. It also generates `tests/conftest.py` with the correct `DYLD_LIBRARY_PATH`.

### 2. Install Python dependencies

```bash
uv sync --group test-integration
```

### 3. Pre-compute GS vectors (optional — pre-computed file included)

The file `ACF_noise_and_covariance_matrix_data/x_and_y_2_sec.pkl` contains GS vectors computed from aLIGO O4 and AdV design PSDs for a 2-second data segment at 4096 Hz. To regenerate (e.g. for a different PSD or duration):

```bash
cd ACF_noise_and_covariance_matrix_data
uv run python Compute_x_and_y.py
```

---

## Running the tests

```bash
# Unit tests only (no LAL required)
uv run --group test pytest tests/test_math.py -v

# Full integration tests (requires lalsuite build)
uv run --group test-integration pytest tests/test_likelihood.py -v

# Everything
uv run --group test-integration pytest tests/ -v
```

107 tests total: 37 unit tests covering GS math and binning geometry, and 70 integration tests covering both exact and relative-binning likelihoods in H1-frame and geocentric-frame configurations.

---

## Running the examples

Both examples expect an injection parameter file as a JSON argument:

```bash
uv run --group test-integration python Examples/PE_relative_binning_22_mode_H1_time.py injection.json
uv run --group test-integration python Examples/Exact_PE.py injection.json
```

The JSON file should contain the injection parameters including `H1_time` (arrival time at LIGO-Hanford). See `Examples/PE_relative_binning_22_mode_H1_time.py` for the full set of expected keys. Both scripts use [bilby](https://github.com/bilby-dev/bilby) + dynesty for sampling.

---

## Notebooks

| Notebook | Description |
|---|---|
| `Comparison_between_relbin_and_exact_log_likelihood.ipynb` | Main validation: relbin vs exact LLR over 500 H1_time samples |
| `Geocent_to_H1_time_conversion.ipynb` | Prior reweighting from geocent_time to H1_time |
| `Time_dependent_antenna_patterns.ipynb` | Timing benchmark of time-dependent antenna patterns |
| `Comparison_between_various_waveform_models.ipynb` | IMRPhenomT variants and mode comparison (requires `gwsurrogate`) |

Run notebooks with:

```bash
uv run --group test-integration jupyter notebook
```

---

## Method summary

The log-likelihood ratio takes the form

$$\ln \mathcal{L}(\theta) - \ln \mathcal{L}_n = \langle d \,|\, h(\theta) \rangle - \frac{1}{2}\langle h(\theta) \,|\, h(\theta) \rangle$$

where the inner product $\langle a \,|\, b \rangle = a^\top C^{-1} b$ is computed via the GS formula. In relative binning, the waveform ratio $r(t) = h_\text{test}(t)/h_\text{fid}(t)$ is approximated as piecewise linear over $N_\text{bins}$ time bins, and the inner products reduce to

$$\langle d \,|\, h \rangle \approx r_0^\top A_0 + r_1^\top A_1, \qquad \langle h \,|\, h \rangle \approx r_0^\top B_0 r_0 + \ldots$$

where $A_0, A_1, B_0$–$B_3$ are the **summary data** pre-computed once from the fiducial waveform and the noise covariance.

---

## Citation

If you use this code, please cite:

```bibtex
@article{sharma2026heterodyning,
  title   = {Rapid inference of gravitational-wave signals in the time domain using a heterodyned likelihood},
  author  = {Sharma, Neha and Vijaykumar, Aditya and Kumar, Prayush},
  journal = {arXiv preprint},
  year    = {2026},
  eprint  = {2601.11239},
  url     = {https://arxiv.org/abs/2601.11239}
}
```

---

## License

See [LICENSE](LICENSE).
