import numpy as np
import os


class Dataset:
    def __init__(self, input_file: str, k_row: int = 0, chi_row: int = 1, experimental: bool = False):
        self._full_path = os.path.abspath(input_file)
        self.name = os.path.splitext(os.path.basename(self._full_path))[0]
        self._k_row = k_row
        self._chi_row = chi_row
        self.is_experimental = experimental
        self._k, self._chi = self.read_data()
        self.k_step = self._k[1] - self._k[0]

        # plotting stuff
        self.color = "#000000"
        self.ls = '-'
        self.lw = 1.5

        # fitting stuff
        self.fix_mix_w = False
        self.mix_w = 1.

    def __repr__(self):
        return self.name

    def read_data(self):
        """
        read data from file and extract k- and chi-column
        :return: tuple of numpy arrays (k, chi)
        """
        data = np.loadtxt(self._full_path)
        k = data[:, self._k_row]
        chi = data[:, self._chi_row]
        return k, chi

    def get_k_chi(self, kw: int, s02: float, k_shift: float):
        """
        Return processed  k-value (apply k-shift): k = k + shift
        and chi-values (multiply chi by s02 coef. and by k power to kw, i.e. k-weight) chi = s02 * chi * k^kw
        :param kw: k-weight, exponent value
        :param s02: magnitude reduction factor
        :param k_shift: shift in k-space
        :return: tuple of numpy arrays (k, chi)
        """
        k = self._k + k_shift if self.is_experimental else self._k
        chi = s02 * self._chi * k ** kw
        return k, chi

    def get_chi(self):
        return self._chi

    def get_k(self):
        return self._k
