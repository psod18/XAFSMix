import os
import tkinter as tk
import numpy as np
import itertools as it
from tkinter import (
    filedialog,
    simpledialog,
)
from utils.custom_frames import (
    PlotWindow,
    ParamsFrame,
    ExperimentalDataFrame,
    ModelDataFrame
)
from utils.dataset import Dataset


class XAFSModelMixerAPI:

    def __init__(self, master):
        self.master = master
        self.master.title("GULP Model Mixer")
        self.master.geometry("750x600")

        # self.data = DataManager()
        self.data: [Dataset] = []

        # plot stuff
        self.k_space_plot = PlotWindow(self, 'k')
        self.r_space_plot = PlotWindow(self, 'r')

        # Frames for global parameter (k-shift, s02, etc.) and plot buttons (k- and R-space
        self.params_frame = ParamsFrame(parent=self.master)
        self.params_frame.grid(row=0, rowspan=2, column=0, padx=10, pady=10, sticky='we')

        # Plot buttons on main frame
        self.plot_k_btn = tk.Button(
            self.master,
            text="Plot k-space",
            command=self.k_space_plot.update_plot,
            relief=tk.RAISED,
            width=25,
        )
        self.plot_k_btn.grid(row=0, column=1, columnspan=3, padx=5, pady=5)

        self.x_min_k = tk.DoubleVar(value=0)
        self.xmin_k = tk.Entry(self.master, width=7, textvariable=self.x_min_k, state=tk.DISABLED,
                               disabledbackground='white', disabledforeground='black')
        self.xmin_k.grid(row=1, column=1, padx=5, pady=5, sticky='we')
        self.xmin_k.bind('<Double-Button-1>', self.set_axis_range)

        tk.Label(self.master, text=': min - max :').grid(row=1, column=2, padx=5, pady=5, sticky='we')
        self.x_max_k = tk.DoubleVar(value=20.0)
        self.xmax_k = tk.Entry(self.master, width=7, textvariable=self.x_max_k, state=tk.DISABLED,
                               disabledbackground='white', disabledforeground='black')
        self.xmax_k.grid(row=1, column=3, padx=5, pady=5, sticky='we')
        self.xmax_k.bind('<Double-Button-1>', self.set_axis_range)

        self.plot_R_btn = tk.Button(
            self.master,
            text="Plot R-space",
            command=self.r_space_plot.update_plot,
            relief=tk.RAISED,
            width=25,
        )
        self.plot_R_btn.grid(row=2, column=1, columnspan=3, padx=5, pady=5)

        self.x_min_r = tk.DoubleVar(value=1.0)
        self.xmin_r = tk.Entry(self.master, width=7, textvariable=self.x_min_r, state=tk.DISABLED,
                               disabledbackground='white', disabledforeground='black')
        self.xmin_r.grid(row=3, column=1, padx=5, pady=5, sticky='we')
        self.xmin_r.bind('<Double-Button-1>', self.set_axis_range)

        tk.Label(self.master, text=': min - max :').grid(row=3, column=2, padx=5, pady=5, sticky='we')
        self.x_max_r = tk.DoubleVar(value=6.0)
        self.xmax_r = tk.Entry(self.master, width=7, textvariable=self.x_max_r, state=tk.DISABLED,
                               disabledbackground='white', disabledforeground='black')
        self.xmax_r.grid(row=3, column=3, padx=5, pady=5, sticky='we')
        self.xmax_r.bind('<Double-Button-1>', self.set_axis_range)

        # experimental frame
        self.experimental_data = ExperimentalDataFrame(gui=self)
        self.experimental_data.grid(row=2, column=0, rowspan=2, padx=10, pady=10, sticky='we')

        self.add_model_btn = tk.Button(
            self.master,
            text="Add model",
            command=self.add_model,
            relief=tk.RAISED,
            width=25,
        )
        self.add_model_btn.grid(row=4, column=0, padx=5, pady=5, sticky='we')

        self.fir_btn = tk.Button(
            self.master,
            text="Fit",
            command=self.fit,
            relief=tk.RAISED,
            width=25,
        )
        self.fir_btn.grid(row=4, column=1, columnspan=3, padx=5, pady=5, sticky='s')

        # last occupied row in main frame
        self.current_row = 4
        self._plot_test = 1

    def set_axis_range(self, event):
        val = simpledialog.askfloat(title="Set axis range", prompt="Set axis value limit")
        if val is not None:
            event.widget.configure(state=tk.NORMAL)
            event.widget.delete(0, tk.END)
            event.widget.insert(0, val)
            event.widget.configure(state=tk.DISABLED)

    def fit(self):
        if self.experimental_data.dataset is None:
            print("No experimental data for fitting procedure")
            # TODO: popup warning
            return False
        if len(self.data) == 1:
            print("No model data for fitting procedure")
            return False

        # get common params
        k_shift = self.params_frame.k_shift_value.get()
        kw = self.params_frame.kw_.get()
        s02 = self.params_frame.s02.get()

        fix_w_data = []
        data_to_fit = []
        w_fix = []
        for ds in self.data:
            if ds.is_experimental:
                k, chi = ds.get_k_chi(kw=kw, s02=1., k_shift=k_shift)
                window = self.window_function(k)
                chi = chi * window
                r_exp, ft_exp = self.calculate_fft(chi=chi, k_step=ds.k_step)
            else:
                if ds.fix_mix_w:
                    w_fix.append(ds.fix_mix_w)
                    fix_w_data.append(ds)
                else:
                    data_to_fit.append(ds)

        fix_w_data = [ds.get_chi() for ds in fix_w_data]
        to_fit = [ds.get_chi() for ds in data_to_fit]

        k_models, dk_step = data_to_fit[-1].get_k(), data_to_fit[-1].k_step  # get k common for all models (ensure,
                                                                             # that all models have identical k )

        weights = self.create_weight_matrix(n_models=len(data_to_fit), max_w=1 - sum(w_fix))
        # TODO: create Custom dict, that have
        weights_r_factor_dict = {}
        for w_set in weights:
            mixed_chi = self.get_weighted_sum(fix_w_models=fix_w_data, fix_weights=w_fix, fit_models=to_fit,
                                              fit_weights=w_set)
            window = self.window_function(k_models)
            mixed_chi = ((s02 * mixed_chi * k_models) ** kw) * window
            r_mod, ft_mod = self.calculate_fft(chi=mixed_chi, k_step=dk_step)
            r_factor = self._calc_chi_squared(exp_data=ft_exp, model=ft_mod)
            weights_r_factor_dict[w_set] = r_factor
            # if len(weights_r_factor_dict) < 10:
            #     weights_r_factor_dict[w_set] = r_factor
            # else:
            #     min_r = min(weights_r_factor_dict.keys(), key=weights_r_factor_dict.__getitem__)
            #     if r_factor < weights_r_factor_dict[min_r]:
            #         del weights_r_factor_dict[min_r]
            #         weights_r_factor_dict[w_set] = r_factor
        for v in weights_r_factor_dict.values():
            print(v)

    @staticmethod
    def get_weighted_sum(fix_w_models, fix_weights, fit_models, fit_weights):
        w = list(fit_weights) + fix_weights
        if fix_w_models:
            mix_chi = (np.vstack([fix_w_models, fit_models]).T * w).T.sum(axis=0)
        else:
            mix_chi = (np.array(fit_models).T * w).T.sum(axis=0)
        return mix_chi

    @staticmethod
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

    @staticmethod
    def _calc_chi_squared(exp_data, model):
        return np.sum((exp_data - model)**2)/np.sum(exp_data**2)

    def delete_data_frame(self, frame: ModelDataFrame):
        self.data.remove(frame.dataset)
        frame.grid_forget()
        frame.destroy()
        self.current_row -= 1

    def import_file(self, invoker):
        filepath = filedialog.askopenfilename(
            filetypes=(
                ("Chi file", "*.chi"),
                ("Data file", "*.dat"),
                ("Text file", "*.txt"),
                ("All files", "*.*")
            )
        )
        if filepath:
            dataset = Dataset(filepath)
            if invoker.dataset is not None:
                self.data.remove(invoker.dataset)
            self.data.append(dataset)
            invoker.dataset = dataset
            invoker.dataset.is_experimental = invoker.experimental
            invoker.line_width.set(dataset.lw)
            if invoker.line_style_menu['state'] == 'disabled':
                invoker.line_style_menu.config(state='normal')
            else:
                invoker.dataset.ls = invoker.line_style.get()
            invoker.name_value.set(os.path.splitext(os.path.basename(filepath))[0])

    def add_model(self):
        filepath = filedialog.askopenfilename(
            filetypes=(
                ("All files", "*.*"),
                ("Chi file", "*.chi"),
                ("Data file", "*.dat"),
                ("Text file", "*.txt"),
            )
        )
        if filepath:
            dataset = Dataset(filepath)
            self.data.append(dataset)
            new_data_frame = ModelDataFrame(gui=self, dataset=dataset)
            new_data_frame.grid(row=self.current_row+1, column=0, padx=5, pady=5, sticky='we')
            new_data_frame.line_width.set(dataset.lw)
            new_data_frame.name_value.set(dataset.name)
            new_data_frame.line_style_menu.config(state='normal')
            self.current_row += 1

    def window_function(self, k):
        k_wind = np.zeros(len(k))
        dx = 1

        # get windows target area from GUI widget
        kmin = self.params_frame.k_min.get()
        kmax = self.params_frame.k_max.get()
        window = self.params_frame.wind.get()
        eps = 0.01

        # get indices of k-values for allocation window function borders
        kmin_ind = np.where(np.abs(k - kmin) < eps)[0][0]
        kmin_ind1 = np.where(np.abs(k - (kmin + dx)) < eps)[0][0]
        kmax_ind = np.where(np.abs(k - kmax) < eps)[0][0]
        kmax_ind1 = np.where(np.abs(k - (kmax - dx)) < eps)[0][0]

        # build window (_/-\_ -> _/|\_ -> _/----\_)
        windows_length = len(k[kmin_ind:kmin_ind1 + 1])

        if window == 'kaiser':
            init_window = np.kaiser(2 * windows_length, 3)
        elif window == 'bartlett':
            init_window = np.bartlett(2 * windows_length)
        elif window == "blackman":
            init_window = np.blackman(2 * windows_length)
        elif window == "hamming":
            init_window = np.hamming(2 * windows_length)
        elif window == "hanning":
            init_window = np.hanning(2 * windows_length)
        else:
            # TODO replace with rectangle window by default?
            raise ValueError("Wrong name for window function")

        max2 = np.where(init_window == max(init_window))[0][1]

        dx1 = [init_window[0:max2]][0]
        dx2 = [init_window[max2:]][0]

        win_shift = int(len(dx1) / 2)

        k_wind[kmin_ind - win_shift:kmin_ind1 - win_shift + 1] = dx1
        k_wind[kmax_ind1 + win_shift:kmax_ind + win_shift + 1] = dx2
        k_wind[kmin_ind1 - win_shift:kmax_ind1 + win_shift] = max(init_window)
        return k_wind

    @staticmethod
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

    def get_data_for_plot(self, space):
        out = []
        # Get necessary params from API entry fields
        k_shift = self.params_frame.k_shift_value.get()
        s02     = self.params_frame.s02.get()
        kw      = self.params_frame.kw_.get()

        if space == 'k':
            # TODO: add possibility to plot windowed data
            for dataset in self.data:
                attr_for_plot = {'label': dataset.name, 'ls': dataset.ls, 'lw': dataset.lw, 'c': dataset.color}
                if dataset.is_experimental:
                    k, chi = dataset.get_k_chi(kw=kw, s02=1., k_shift=k_shift)
                else:
                    k, chi = dataset.get_k_chi(kw=kw, s02=s02, k_shift=0)
                out.append([k, chi, attr_for_plot])

        elif space == 'r':
            for dataset in self.data:
                attr_for_plot = {'label': dataset.name, 'ls': dataset.ls, 'lw': dataset.lw, 'c': dataset.color}
                if dataset.is_experimental:
                    k, chi = dataset.get_k_chi(kw=kw, s02=1., k_shift=k_shift)
                else:
                    k, chi = dataset.get_k_chi(kw=kw, s02=s02, k_shift=0)
                window = self.window_function(k)
                chi = chi * window
                r, ft = self.calculate_fft(chi=chi, k_step=dataset.k_step)
                out.append([r, ft, attr_for_plot])
        else:
            print(f"Unknown space {space}")

        # TODO: add fit plot to plotting data
        return out


if __name__ == "__main__":
    root = tk.Tk()
    api = XAFSModelMixerAPI(root)
    root.mainloop()
