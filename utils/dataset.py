import numpy as np
import os


class Dataset:
    def __init__(self, input_file: str, k_row: int = 0, chi_row: int = 1, experimental: bool = False):
        self._full_path = os.path.abspath(input_file)
        self.name = os.path.splitext(os.path.basename(self._full_path))[0]
        self._k_row = k_row
        self._chi_row = chi_row
        self.is_experimental = experimental
        self.k, self.chi = self.read_data()
        self.k_step = self.k[1] - self.k[0]

        # plotting stuff
        self.color = "#000000"
        self.ls = '-'
        self.lw = 1.5

    def __repr__(self):
        return self.name

    def read_data(self):
        data = np.loadtxt(self._full_path)
        _k = data[:, self._k_row]
        _chi = data[:, self._chi_row]
        return _k, _chi
