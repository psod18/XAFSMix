import os
import tkinter as tk
import numpy as np
import itertools as it
from tkinter import (
    colorchooser,
    filedialog,
    simpledialog,
)
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg,
    NavigationToolbar2Tk,
)
from scipy.optimize import curve_fit

from utils.dataset import Dataset


class ParamsFrame(tk.Frame):
    def __init__(self, parent):
        super().__init__(parent, border=2, relief=tk.GROOVE)
        tk.Label(self, text="k-shift: ").grid(row=0, column=0, padx=5, pady=5, sticky='e')
        self.k_shift_value = tk.DoubleVar(value=0.0)
        self.kshift = tk.Entry(self, width=7, textvariable=self.k_shift_value, state=tk.DISABLED,
                               disabledbackground='white', disabledforeground='black')
        self.kshift.grid(row=0, column=1, padx=5, pady=5)
        self.kshift.bind('<Double-Button-1>', self.set_k_shift)

        # first row
        tk.Label(self, text="s02: ").grid(row=0, column=2, padx=5, pady=5, sticky='e')
        self.s02 = tk.DoubleVar(value=1.)
        self.mag = tk.Entry(self, width=7, textvariable=self.s02, state=tk.DISABLED,
                            disabledbackground='white', disabledforeground='black')
        self.mag.grid(row=0, column=3, padx=5, pady=5)
        self.mag.bind('<Double-Button-1>', self.set_magnitude)

        # Create window function selector
        tk.Label(self, text="Window:").grid(row=0, column=4, padx=5, pady=5)
        self.wind = tk.StringVar(self)
        choices = ('kaiser', 'bartlett', 'blackman', 'hamming', 'hanning')
        self.wind.set(choices[0])  # set the default option
        self.wind_func_menu = tk.OptionMenu(self, self.wind, *choices)
        self.wind_func_menu.grid(row=0, column=5, padx=5, pady=5, sticky="ew")
        self.wind_func_menu.configure(width=10)

        # second row:
        tk.Label(self, text="k min: ").grid(row=1, column=0, padx=5, pady=5, sticky='e')
        self.k_min = tk.DoubleVar(value=3.5)
        self.kmin = tk.Entry(self, width=7, textvariable=self.k_min, state=tk.DISABLED,
                             disabledbackground='white', disabledforeground='black')
        self.kmin.grid(row=1, column=1, padx=5, pady=5)
        self.kmin.bind('<Double-Button-1>', self.set_k_min)

        tk.Label(self, text="k max: ").grid(row=1, column=2, padx=5, pady=5, sticky='e')
        self.k_max = tk.DoubleVar(value=12.0)
        self.kmax = tk.Entry(self, width=7, textvariable=self.k_max, state=tk.DISABLED,
                             disabledbackground='white', disabledforeground='black')
        self.kmax.grid(row=1, column=3, padx=5, pady=5)
        self.kmax.bind('<Double-Button-1>', self.set_k_max)

        tk.Label(self, text="k-weight:").grid(row=1, column=4, padx=5, pady=5, sticky='e')
        self.kw_ = tk.IntVar(self)
        weights_ = [i for i in range(5)]
        self.kw_.set(weights_[1])  # set the default option
        self.kw_menu = tk.OptionMenu(self, self.kw_, *weights_)
        self.kw_menu.grid(row=1, column=5, padx=5, pady=5, sticky="ew")
        self.kw_menu.configure(width=10)

    def set_k_shift(self, *args):
        new_k_shift = simpledialog.askfloat(title="k-shift", prompt="Set value for data shift in k-space",
                                            minvalue=-10, maxvalue=10)
        if new_k_shift is not None:
            self.k_shift_value.set(new_k_shift)

    def set_magnitude(self, *args):
        new_mag = simpledialog.askfloat(title="Magnitude coeff.", prompt="Set value for signal magnitude coefficient",
                                        minvalue=0.01, maxvalue=1.)
        if new_mag:
            self.s02.set(new_mag)

    def set_k_min(self, *args):
        new_kmin = simpledialog.askfloat(title="Window func. k-min", prompt="Set lowest bound  value for window "
                                                                            "function",
                                         minvalue=0, maxvalue=self.k_max.get())
        if new_kmin is not None:
            self.k_min.set(new_kmin)

    def set_k_max(self, *args):
        new_kmax = simpledialog.askfloat(title="Window func. k-max", prompt="Set highest bound  value for window "
                                                                            "function",
                                         minvalue=self.k_min.get(), maxvalue=20)
        if new_kmax is not None:
            self.k_max.set(new_kmax)


class BaseDataFrame(tk.Frame):
    def __init__(self, gui):
        super().__init__(gui.master, border=2, relief=tk.GROOVE)
        self.gui = gui
        self.dataset: [Dataset, None] = None

        self.name_value = tk.StringVar()
        self.name = tk.Entry(self, width=30, textvariable=self.name_value, disabledbackground='white',
                             disabledforeground='black')
        self.name.grid(row=0, column=0, columnspan=2, padx=5, pady=5, stick='we')
        self.name.config(state=tk.DISABLED)
        self.name.bind('<Double-Button-1>', self.rename_dataset)

        tk.Label(self, text="line width:").grid(row=1, column=2, padx=5, pady=5, sticky='e')
        self.line_width = tk.DoubleVar(self)
        self.line_width_entry = tk.Entry(self, width=7, textvariable=self.line_width, disabledbackground='white',
                                         disabledforeground='black')
        self.line_width_entry.grid(row=1, column=3, padx=5, pady=5, sticky="ew")
        self.line_width_entry.config(state=tk.DISABLED)
        self.line_width_entry.bind('<Double-Button-1>', self.change_line_width)

        tk.Label(self, text="line style:").grid(row=1, column=4, padx=5, pady=5, sticky='e')
        self.line_style = tk.StringVar(self)
        choices = ('solid', 'dashed', 'dashdot', 'dotted')
        self.line_style.set(choices[0])  # set the default option
        self.line_style_menu = tk.OptionMenu(self, self.line_style, *choices)
        self.line_style_menu.grid(row=1, column=5, padx=5, pady=5, sticky="ew")
        self.line_style_menu.configure(width=7, state='disable')
        self.line_style.trace_add('write', self.change_line_style)

        self.color = tk.Label(self, text="", bg='black', width=3)
        self.color.grid(row=1, column=6, padx=5, pady=5, sticky='we')
        self.color.bind('<Double-Button-1>', self.change_color)

    def rename_dataset(self, *args):
        if self.dataset:
            current_name = self.name_value.get()
            new_name = simpledialog.askstring('Change data label', 'Change name', initialvalue=current_name)
            if new_name and new_name != current_name:
                ds_idx = self.gui.data.index(self.dataset)
                self.name_value.set(new_name)
                self.dataset.name = new_name
                self.gui.data[ds_idx] = self.dataset

    def change_line_width(self, *args):
        if self.dataset:
            lw = simpledialog.askfloat(title="Line width", prompt="Set curve line width", minvalue=0.1, maxvalue=10)
            if lw:
                self.line_width.set(lw)
                self.dataset.lw = lw

    def change_color(self, *args):
        if self.dataset:
            new_color = colorchooser.askcolor()[-1]  # take only in hex-form
            if new_color:
                self.dataset.color = new_color
                self.color.config(bg=new_color)

    def change_line_style(self, *args):
        if self.dataset:
            new_ls = self.line_style.get()
            self.dataset.ls = new_ls


class ExperimentalDataFrame(BaseDataFrame):
    def __init__(self, gui):
        super().__init__(gui=gui)
        self.gui = gui
        self.experimental = True
        # self.dataset: Dataset = None
        self.name_value.set("No imported Data")
        tk.Button(
            self,
            text="Import Experimental data",
            relief=tk.RAISED,
            command=lambda: self.gui.import_file(self),
            width=25,
        ).grid(row=1, column=0, columnspan=2, padx=5, pady=5, stick='we')


class ModelDataFrame(BaseDataFrame):
    def __init__(self, gui, dataset):

        super().__init__(gui=gui)
        self.gui = gui
        self.experimental = True
        self.dataset: Dataset = dataset

        # open data button
        tk.Button(
            self,
            text="...",
            relief=tk.RAISED,
            command=lambda: self.gui.import_file(self),
            width=10,
        ).grid(row=1, column=0, padx=5, pady=5, stick='we')

        # remove frame
        tk.Button(
            self,
            text="X",
            relief=tk.RAISED,
            command=lambda: self.gui.delete_data_frame(self),
            width=10,
            fg='red',
        ).grid(row=1, column=1, padx=5, pady=5, stick='we')

        tk.Label(self, text="weight: ").grid(row=0, column=2, padx=5, pady=5, sticky='e')
        self.w = tk.DoubleVar(value=0.0)
        self.weight = tk.Entry(self, width=7, textvariable=self.w)
        self.weight.grid(row=0, column=3, padx=5, pady=5, stick='we')

        # self.hold_w_value = tk.BooleanVar()
        # tk.Checkbutton(self, text="Freeze weight", variable=self.hold_w_value).grid(row=0, column=4, stick='sn')


class PlotWindow:

    def __init__(self, parent, space: str):
        self.parent: XAFSModelMixerAPI = parent
        self.window = None
        self.space = space
        self.canvas = None
        self.fig = None

    def build_plot_window(self):
        self.window = tk.Toplevel(self.parent.master)
        self.window.title(f"{self.space}-space")
        self.window.geometry("800x600")
        self.fig = plt.Figure()
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.window)
        toolbar = NavigationToolbar2Tk(self.canvas, self.window)
        toolbar.update()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1.0)
        self.window.protocol("WM_DELETE_WINDOW", self._on_closing_plot)

    def _on_closing_plot(self):
        self.window.destroy()
        self.canvas = None
        self.fig = None

    def update_plot(self, *args):
        # pp = self.parent.collect_params()
        if self.canvas is None:
            self.build_plot_window()
        if self.fig.get_axes():
            ax = self.fig.get_axes()[0]
            ax.cla()
        else:
            ax = self.fig.add_subplot(111)
        if self.space == 'k':
            kw = self.parent.params_frame.kw_.get()
            # build label for y-axis in k-space
            y_label = r'$\chi$(k)' if kw == 0 else r'k$^{}{}{}\cdot\chi$(k)'.format('{', kw, '}')

            ax.set_xlabel(r'k, $\AA^{-1}$')
            ax.set_ylabel(y_label)
            ax.set_xlim(self.parent.x_min_k.get(), self.parent.x_max_k.get())
            ax.set_ylim(auto=True)
        elif self.space == 'r':
            ax.set_xlabel(r'R, $\AA$')
            ax.set_ylabel(r'|FT($\chi$)|')
            ax.set_xlim(self.parent.x_min_r.get(), self.parent.x_max_r.get())
            ax.set_ylim(auto=True)
        for ds in self.parent.get_data_for_plot(self.space):
            x, y, attr = ds
            ax.plot(x, y, **attr)

        if self.parent.data:
            ax.legend()
        ax.grid(True)
        self.canvas.draw()


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
            mixed_chi = (s02 * mixed_chi * k_models ** kw) * window
            r_mod, ft_mod = self.calculate_fft(chi=mixed_chi, k_step=dk_step)
            r_factor = self._calc_chi_squared(exp_data=ft_exp, model=ft_mod)
            if len(weights_r_factor_dict) < 10:
                weights_r_factor_dict[w_set] = r_factor
            else:
                min_r = min(weights_r_factor_dict.keys(), key=weights_r_factor_dict.__getitem__)
                if r_factor < weights_r_factor_dict[min_r]:
                    del weights_r_factor_dict[min_r]
                    weights_r_factor_dict[w_set] = r_factor
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
