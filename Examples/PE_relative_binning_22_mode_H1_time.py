#!/usr/bin/env python
# coding: utf-8
import sys
import os
import numpy as np
from pycbc.detector import Detector
import bilby
import json
import dill
from pycbc.waveform import get_td_waveform
from bilby.gw.conversion import component_masses_to_chirp_mass, chirp_mass_and_mass_ratio_to_component_masses
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(_REPO_ROOT, 'Relative_binning_class'))
sys.path.insert(0, os.path.join(_REPO_ROOT, 'ACF_noise_and_covariance_matrix_data'))
from Relative_22_H1_frame import *
# Same script can be used to sample geocentric time as well. The priors need to be modified.
# from Noise_realisation import noise_realisation_from_psd  # only needed for Gaussian noise block below

# Loading the arguments from submit file.
dict_file = sys.argv[1]

with open(dict_file, "r") as f:
    injection_parameters = json.load(f)   # Load a .json file or write the dictionaries of injection parameters here.

"""
H1_ifo = bilby.gw.detector.get_empty_interferometer("H1")
to_h1 = H1_ifo.time_delay_from_geocenter(
    ra = injection_parameters["ra"],
    dec = injection_parameters["dec"],
    time = injection_parameters["geocent_time"]
    )
injection_parameters["H1_time"] = injection_parameters["geocent_time"] + to_h1
injection_parameters.pop("geocent_time")
"""

fiducial_parameters = injection_parameters.copy()

n = 2 #float(sys.argv[2])
epsilon1_H1, epsilon2_H1 = 0.5,0.1 #float(sys.argv[3]),float(sys.argv[4])
epsilon1_L1, epsilon2_L1 = 0.5,0.1 #float(sys.argv[5]),float(sys.argv[6])
epsilon1_V1, epsilon2_V1 = 0.5,0.1 #float(sys.argv[7]),float(sys.argv[8])

spacing1, spacing2, spacing3,spacing_times1,spacing_times2 = 1, 1, 100,0.02, 0.05 #int(sys.argv[9]),int(sys.argv[10]),int(sys.argv[11]), float(sys.argv[12]),float(sys.argv[13])

spacing = [spacing1, spacing2, spacing3]
epsilon = {"H1":np.array([epsilon1_H1,epsilon2_H1]),"L1":np.array([epsilon1_L1,epsilon2_L1]),"V1":np.array([epsilon1_V1,epsilon2_V1])}
spacing_times = [spacing_times1, spacing_times2]
print("n,epsilon,spacing,spacing_times", n, epsilon, spacing,spacing_times)

# Defining the time array:

Time_array = np.arange(-n + 0.5, 0.5, 1 / 4096.0, dtype=np.float64)
print("t[0]:", Time_array[0])

with open(os.path.join(_REPO_ROOT, "ACF_noise_and_covariance_matrix_data", "x_and_y_2_sec.pkl"), "rb") as f:
    Inv_cov_Noise = dill.load(f)
x = Inv_cov_Noise["x"]
y = Inv_cov_Noise["y"]

Detectors_list = {"H1":Detector("H1"),"L1":Detector("L1"),"V1":Detector("V1")}
Noise= {"H1":0,"L1":0,"V1":0}
print("Noise : ", Noise)
Data_list = {}

# If user wants to use gaussian noise instead of zero noise:
"""
Detectors_list_for_noise = {"L1":bilby.gw.detector.get_empty_interferometer("L1"),"H1":bilby.gw.detector.get_empty_interferometer("H1"),"V1":bilby.gw.detector.get_empty_interferometer("V1")}
Noise = noise_realisation_from_psd(length_of_noise_segment=int(n), Detectors_list=Detectors_list_for_noise, sampling_frequency=4096)
"""

Likelihood_obj = RelativeBinningTimeDomainH1detectorframe(
    time = Time_array,
    Detectors_list = Detectors_list,
    fiducial_parameters = fiducial_parameters,
    injection_parameters = injection_parameters,
    Data_list = Data_list,
    x = x,
    y = y,
    Noise = Noise,
    fmin=10,
    fref=20,
    epsilon_array_per_detector=epsilon,
    spacing=spacing,
    spacing_times=spacing_times
)
SNR, Network_SNR = Likelihood_obj.compute_SNR_TD_and_waveform_data()[0:2]

# Defining Priors:
# This way of defining priors is entirely my choice. A more general approach would to be use prior files available in 'bilby'.

del_mc = (
    (1.2e-4) * injection_parameters["chirp_mass"] ** (8 / 3) * (10 / Network_SNR)*5
)  # Using https://arxiv.org/pdf/gr-qc/9402014 (with extra factor of 100)

if del_mc < 2.0:
    del_mc = 2.0
mc_min = injection_parameters["chirp_mass"] - del_mc
mc_max = injection_parameters["chirp_mass"] + del_mc

priors = bilby.gw.prior.CBCPriorDict(
    dict(
        chirp_mass=bilby.core.prior.Uniform(
            minimum=mc_min,
            maximum=mc_max,
            name="chirp_mass",
            latex_label="$\\mathcal{M}$",
        ),
        mass_ratio=bilby.core.prior.Uniform(
            minimum=1 / 6,
            maximum=1.0,
            name="mass_ratio",
            latex_label="$q$",
        ),
        chi_1=bilby.core.prior.Uniform(
            minimum=-0.99,
            maximum=0.99,
            name="chi_1",
            latex_label="$\\chi_1$",
        ),
        chi_2=bilby.core.prior.Uniform(
            minimum=-0.99,
            maximum=0.99,
            name="chi_2",
            latex_label="$\\chi_2$",
        ),
        luminosity_distance=bilby.gw.prior.UniformSourceFrame(
            name="luminosity_distance", minimum=10, maximum=5000, unit="Mpc"
        ),
        theta_jn=bilby.core.prior.Sine(name="theta_jn"),
        ra=bilby.core.prior.Uniform(
            minimum=0, maximum=2 * np.pi, name="ra", boundary="periodic"
        ),
        dec=bilby.core.prior.Cosine(name="dec"),
        psi=bilby.core.prior.Uniform(
            minimum=0, maximum=np.pi, name="psi", boundary="periodic"
        ),
        phase=bilby.core.prior.Uniform(
            minimum=0, maximum=2 * np.pi, name="phase", boundary="periodic"
        ),
        H1_time=bilby.core.prior.Uniform(
            injection_parameters["H1_time"] - 0.01,
            injection_parameters["H1_time"] + 0.01,
            name="H1_time",
            latex_label="$t_{H1}$",
            unit=None,
            boundary=None,
        ),
    )
)

# The addition of noise shifts the posterior. In this case, the fiducial parameters same as injection parameters may not be the best choice. Thus, we choose fiducial parameters using scipy.optimize:

"""
Maximum_likelihood_parameters_obj = Set_Fiducial_parameters(
    time = Time_array,
    Detectors_list = Detectors_list,
    fiducial_parameters = fiducial_parameters,
    injection_parameters = injection_parameters,
    Data_list = Data_list,
    x = x,
    y = y,
    Noise = Noise,
    priors = priors,
    fmin=10,
    fref=20,
)

new_fiducial_parameters = (
    Maximum_likelihood_parameters_obj.optimize_fiducial_parameters()
)
fiducial_parameters = new_fiducial_parameters.copy()

del Maximum_likelihood_parameters_obj
del new_fiducial_parameters
del Likelihood_obj

Likelihood_obj = RelativeBinningTimeDomainH1detectorframe(
    time = Time_array,
    Detectors_list = Detectors_list,
    fiducial_parameters = fiducial_parameters,
    injection_parameters = injection_parameters,
    Data_list = Data_list,
    x = x,
    y = y,
    Noise = Noise,
    fmin=10,
    fref=20,
    epsilon_array_per_detector=epsilon,
    spacing=spacing,
    spacing_times=spacing_times
)

"""
SNR, Network_SNR = Likelihood_obj.compute_SNR_TD_and_waveform_data()[0:2]
print("SNR, Network SNR : ", SNR, Network_SNR)
print("Fiducial_parameters : ", Likelihood_obj.fiducial_parameters)
Likelihood_obj.parameters.update(fiducial_parameters)
print("Fiducial Likelihood : ", Likelihood_obj.log_likelihood_ratio())
print("Injection_parameters : ", Likelihood_obj.injection_parameters)
print("Number of bins : ",Likelihood_obj.number_of_bins)
print("Fiducial waveform : ", Likelihood_obj.h22_fiducial)

# We can redefine the priors on chirp mass for a new choice of fiducial parameters. This is optional, but if the prior is too small, using an older version of prior may not recover full posterior.
"""
mc_min = fiducial_parameters["chirp_mass"] - del_mc
mc_max = fiducial_parameters["chirp_mass"] + del_mc

priors["chirp_mass"] = bilby.core.prior.Uniform(
    minimum=mc_min,
    maximum=mc_max,
    name="chirp_mass",
    latex_label="$\\mathcal{M}$",
)
"""
parameters_to_sample = [
    "chirp_mass",
    "mass_ratio",
    "chi_1",
    "chi_2",
    "luminosity_distance",
    "theta_jn",
    "ra",
    "dec",
    "psi",
    "phase",
    "H1_time",
]

for key in injection_parameters:
    if key not in parameters_to_sample:
        priors[key] = injection_parameters[key]

# Sampling:
fname = "Relative_binning"
foldername = "Test_1"

result = bilby.run_sampler(
    likelihood=Likelihood_obj,
    priors=priors,
    nlive=500,
    label=fname,
    outdir=foldername,
    sampler="dynesty",
    sample="acceptance-walk",
    naccept=60,
    injection_parameters=injection_parameters,
    npool=16
)
