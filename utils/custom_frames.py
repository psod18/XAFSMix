import tkinter as tk
from tkinter import (
    colorchooser,
    simpledialog,
)
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg,
    NavigationToolbar2Tk,
)


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

