import datetime
import tkinter as tk
from tkinter import (
    colorchooser,
    simpledialog,
    messagebox,
)
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg,
    NavigationToolbar2Tk,
)


from utils.dataset import Dataset
from utils.tools import (
    calculate_fft,
    create_weight_matrix,
    get_weighted_sum,
    calc_chi_squared,
)
from utils.validators import (
    check_r_factor_value,
    enable_entry_editing,
)


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
        self.experimental = False
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
        self.weight = tk.DoubleVar(value=0.0)
        # self.weight.trace("w", self.set_model_weight_to_dataset)
        self.weight_entry = tk.Entry(self, width=7, textvariable=self.weight, state=tk.DISABLED)
        self.weight_entry.grid(row=0, column=3, padx=5, pady=5, stick='we')
        self.weight_entry.bind('<Return>', self.set_model_weight_to_dataset)
        self.weight_entry.bind('<FocusOut>', self.set_model_weight_to_dataset)

        self.hold_w = tk.BooleanVar(value=False)
        tk.Checkbutton(self, text="Hold weight", variable=self.hold_w).grid(row=0, columnspan=2, column=4, stick='sn')
        self.hold_w.trace("w", self.set_model_weight_in_widget)

    def set_model_weight_in_widget(self, *args):
        if self.hold_w.get():
            self.weight_entry.config(state='normal')
            self.dataset.fix_mix_w = True
        else:
            self.weight.set(0.0)
            self.weight_entry.config(state='disabled')
            self.dataset.fix_mix_w = False

    def set_model_weight_to_dataset(self, *args):
        try:
            val = float(self.weight_entry.get())
            if 1. > val > 0.:
                self.weight.set(val)
            else:
                self.weight.set(0.0)
        except ValueError:
            self.weight.set(0.0)
        self.dataset.mix_w = self.weight.get()


class PlotWindow:

    def __init__(self, parent, space: str):
        self.parent = parent
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

    def update_plot(self, dataset):
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
        for ds in dataset:
            x, y, attr = ds
            ax.plot(x, y, **attr)

        if self.parent.data:
            ax.legend()
        ax.grid(True)
        self.canvas.draw()


class FittingFrame(tk.Frame):

    def __init__(self, gui):
        super().__init__(gui.master)
        self.gui = gui
        self.fit_dataset = []
        self.fix_dataset = []
        self.data_r_space = {}
        self.data_k_space = {}
        self.weights = None

        # DD menu and label holders:
        self.model_mix_window = None

        self.fit_btn = tk.Button(
            self,
            text="Fit",
            command=self.fit,
            relief=tk.RAISED,
            width=25,
        )
        self.fit_btn.grid(row=0, column=0, columnspan=2, padx=5, pady=5, sticky='we')

        tk.Label(self, text="R-factor threshold:").grid(row=1, column=0, padx=5, pady=5, sticky='we')
        self.r_factor_th = tk.DoubleVar(value=.02)
        self.threshold = tk.Entry(self, textvariable=self.r_factor_th, width=7,
                                  state = tk.DISABLED, disabledbackground = 'white',
                                  disabledforeground = 'black'
                                    )
        self.threshold.grid(row=1, column=1, padx=5, pady=5, sticky='w')
        self.threshold.bind('<Return>', check_r_factor_value)
        self.threshold.bind('<FocusOut>', check_r_factor_value)
        self.threshold.bind('<Double-Button-1>', enable_entry_editing)

    def build_mix_dropdown_menu(self):
        if len(self.data_r_space) > 1:
            self.model_mix_window = tk.Toplevel(self)
            self.model_mix_window.title("Model mixing and Fitting results")
            tk.Label(self.model_mix_window, text="R-factor and weights").grid(row=0, column=0, padx=5, pady=5,
                                                                              sticky='wse')

            self.model_mix_window.r_factor = tk.StringVar(self.model_mix_window)
            choices = tuple(sorted([k for k in self.data_r_space.keys()][1:]))
            self.model_mix_window.r_factor.trace_add('write', self.plot_mix_vs_exp)
            self.model_mix_window.r_factor.set(choices[0])  # set the default option
            self.model_mix_window.model_mix = tk.OptionMenu(self.model_mix_window, self.model_mix_window.r_factor, *choices)
            self.model_mix_window.model_mix.grid(row=1, column=0, padx=5, pady=5, sticky="ew")
            self.model_mix_window.model_mix.configure(width=50)

            self.model_mix_window.protocol("WM_DELETE_WINDOW", self.destroy_fit_window)
        else:
            messagebox.showwarning(
                title="No satisfying R-factor",
                message="Any weighted mix doe not provide low R-factor",
            )
            self.destroy_fit_window()

    def destroy_fit_window(self):
        if self.model_mix_window is not None:
            self.model_mix_window.destroy()
            self.model_mix_window.update()
            self.model_mix_window = None
        self.fit_dataset.clear()
        self.fix_dataset.clear()
        self.data_r_space.clear()
        self.data_k_space.clear()

    def plot_mix_vs_exp(self, *args):

        mix_key = self.model_mix_window.r_factor.get()
        data_r = self.data_r_space['exp'], self.data_r_space[mix_key]
        data_k = self.data_k_space['exp'], self.data_k_space[mix_key]

        self.gui.k_space_fit_plot.update_plot(dataset=data_k)
        self.gui.r_space_fit_plot.update_plot(dataset=data_r)

    def fit(self):
        self.destroy_fit_window()
        if self.gui.experimental_data.dataset is None:
            messagebox.showwarning(
                title="Experimental data not found",
                message="Cannot execute fitting procedure - experimental dataset was not imported.",
            )
            return False

        if len(self.gui.data) == 1:
            messagebox.showwarning(
                title="Lack of theoretical dataset(s)",
                message="Cannot execute fitting procedure - any theoretical dataset was not add.",
            )
            return False
        # TODO: add messagebox for only ine theoretical dataset (messagebox.askyesno(title=None, message=None))

        # get common params
        k_shift = self.gui.params_frame.k_shift_value.get()
        kw = self.gui.params_frame.kw_.get()
        s02 = self.gui.params_frame.s02.get()
        min_r = self.gui.x_min_r.get()
        max_r = self.gui.x_max_r.get()

        w_fix = []

        for ds in self.gui.data:
            if ds.is_experimental:
                k, chi = ds.get_k_chi(kw=kw, s02=1., k_shift=k_shift)
                self.data_k_space['exp'] = (k, chi, {'label': ds.name, 'ls': ds.ls, 'lw': ds.lw, 'c': 'k'})
                window = self.gui.window_function(k)
                chi = chi * window
                r_exp, ft_exp = calculate_fft(chi=chi, k_step=ds.k_step)
                self.data_r_space['exp'] = (r_exp, ft_exp, {'label': ds.name, 'ls': ds.ls, 'lw': ds.lw, 'c': 'k'})
            else:
                if ds.fix_mix_w:
                    w_fix.append(ds.mix_w)
                    self.fix_dataset.append(ds)
                else:
                    self.fit_dataset.append(ds)

        min_r_idx = np.where(r_exp > min_r)[0][0]
        max_r_idx = np.where(r_exp > max_r)[0][0]

        fixed_weight_data = [ds.get_chi() for ds in self.fix_dataset]
        to_fit = [ds.get_chi() for ds in self.fit_dataset]
        # TODO: merge k for all models
        k_models, dk_step = self.fit_dataset[-1].get_k(), self.fit_dataset[-1].k_step

        t = datetime.datetime.now()
        if self.weights is None or len(self.weights[0]) != len(to_fit):
            self.weights = create_weight_matrix(n_models=len(self.fit_dataset), max_w=1 - sum(w_fix))

        for w_set in self.weights:
            mixed_chi = get_weighted_sum(fix_w_models=fixed_weight_data,
                                         fix_weights=w_fix,
                                         fit_models=to_fit,
                                         fit_weights=w_set,
                                         )
            window = self.gui.window_function(k_models)
            mixed_chi = ((s02 * mixed_chi) * k_models ** kw)

            mixed_chi_windowed = mixed_chi * window

            r_mod, ft_mod = calculate_fft(chi=mixed_chi_windowed, k_step=dk_step)
            r_factor = calc_chi_squared(exp_data=ft_exp, model=ft_mod, min_r=min_r_idx, max_r=max_r_idx)

            if r_factor <= self.r_factor_th.get():
                label = self.build_label(w_set, r_factor)
                attr_dict = {'label': label, 'lw': 2, 'c': 'r', 'marker': 'o', 'markersize': 9, 'mfc': 'none'}
                key = f"{r_factor:.5f}: " + " ".join([str(i) for i in w_set])
                self.data_k_space[key] = (k_models, mixed_chi, attr_dict)
                self.data_r_space[key] = (r_mod, ft_mod, attr_dict)

        self.build_mix_dropdown_menu()
        # return True

    def build_label(self, weights, r_factor):
        label = f"R-factor: {r_factor:.5f}\n"
        names = [ds.name for ds in self.fit_dataset]
        for name, percent in zip(names, weights):
            label += f"{name}: {percent}\n"

        for ds in self.fix_dataset:
            label += f"{ds.name}: {ds.mix_w} (fix)\n"
        return label.strip()
