from typing import List

import numpy as np

from utils.dataset import Dataset
from utils.product import coefficient_combiner


def calculate_fft(chi, k_step, n_samples: int = 2048):
    """
    Perform Fourier Transform with dataset
    :param chi: chi-data - must be windowed already AND:
        (1) weighted with s02 [if theoretical]
        (2) k-shifted [if experimental]
        (3) weighted [if for fitting and not experimental]
    :param k_step: step size in k-space
    :param n_samples:
    :return: r and sqrt(Im^2 + Re^2)
    """

    rstep = np.pi / (k_step * n_samples)

    ft_chi = np.fft.fft(chi, n=n_samples) * (k_step / np.sqrt(np.pi))

    rmax_index = n_samples // 2
    ft_mod = np.sqrt(ft_chi.real ** 2 + ft_chi.imag ** 2)
    r = rstep * np.arange(rmax_index)
    return r[:rmax_index], ft_mod[:rmax_index]


def calc_chi_squared(exp_data, model, min_r, max_r):
    return np.sum((exp_data[min_r:max_r] - model[min_r:max_r])**2)/np.sum(exp_data[min_r:max_r]**2)


def create_weight_matrix(n_models: int, max_w: float) -> np.array:
    _weights = []
    for w in coefficient_combiner(n_models, max_w, weight_step=0.01):
        _weights.append(w)
    return np.array(_weights)


def adjust_spectra_over_k_range(models: List[Dataset]):
    _k = np.vstack([m.get_k() for m in models])
    _, init_idx = np.where(0.03 > np.abs(_k - _k[np.argmax(_k[:, 0]), 0]))
    _, final_idx = np.where(0.03 > np.abs(_k - _k[np.argmin(_k[:, -1]), -1]))
    k = _k[0, init_idx[0]:final_idx[0]+1]

    chi = np.vstack([model.get_chi()[i:j+1] for model, i, j in zip(models, init_idx, final_idx)])
    return chi, k
