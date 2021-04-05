import numpy as np
import itertools as it


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


def create_weight_matrix(n_models: int, max_w: float):
    _weights = []
    step = 100
    eps = 5 / (step * 10)
    w = [i / step for i in range(int(max_w*step) + 1)]
    prod = it.product(w, repeat=n_models)

    for i in prod:
        if max_w - eps <= sum(i) <= max_w + eps:
            _weights.append(i)
    return _weights


def get_weighted_sum(fix_w_models, fix_weights, fit_models, fit_weights):
    w = list(fit_weights) + fix_weights
    if fix_w_models:
        mix_chi = (np.vstack([fix_w_models, fit_models]).T * w).T.sum(axis=0)
    else:
        mix_chi = (np.array(fit_models).T * w).T.sum(axis=0)
    return mix_chi

