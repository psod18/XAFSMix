import os
import tkinter as tk
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
from utils.dataset import (
    Dataset,
    DataManager,
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
        tk.Label(self, text="Mag.: ").grid(row=0, column=2, padx=5, pady=5, sticky='e')
        self.s02 = tk.DoubleVar(value=0.81)
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
        if new_k_shift:
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
        if new_kmin:
            self.k_min.set(new_kmin)

    def set_k_max(self, *args):
        new_kmax = simpledialog.askfloat(title="Window func. k-max", prompt="Set highest bound  value for window "
                                                                            "function",
                                         minvalue=self.k_min.get(), maxvalue=20)
        if new_kmax:
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
            print(new_ls)
            self.dataset.ls = new_ls


class ExperimentalDataFrame(BaseDataFrame):
    def __init__(self, gui):
        super().__init__(gui=gui)
        self.gui = gui
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
        # self.fig.add_subplot(111)
        self.window.protocol("WM_DELETE_WINDOW", self._on_closing_plot)

    def _on_closing_plot(self):
        self.window.destroy()
        self.canvas = None
        self.fig = None

    def update_plot(self):
        if self.canvas is None:
            self.build_plot_window()
        if self.fig.get_axes():
            ax = self.fig.get_axes()[0]
            ax.cla()
        else:
            ax = self.fig.add_subplot(111)
        for ds in self.parent.data.get_data_for_plot(self.space, **self.parent.collect_params()):
            x, y, attr = ds
            ax.plot(x, y, **attr)
        if self.fig.get_axes():
            ax.legend()
        self.canvas.draw()


class XAFSModelMixerAPI:

    def __init__(self, master):
        self.master = master
        self.master.title("GULP Model Mixer")
        self.master.geometry("800x600")

        self.data = DataManager()

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
        self.plot_k_btn.grid(column=1, row=0, padx=5, pady=5)

        self.plot_R_btn = tk.Button(
            self.master,
            text="Plot R-space",
            command=self.r_space_plot.update_plot,
            relief=tk.RAISED,
            width=25,
        )
        self.plot_R_btn.grid(row=1, column=1, padx=5, pady=5)

        # experimental frame
        self.experimental_data = ExperimentalDataFrame(gui=self)
        self.experimental_data.grid(row=2, column=0, padx=10, pady=10, sticky='we')

        self.add_model_btn = tk.Button(
            self.master,
            text="Add data",
            command=self.add_model,
            relief=tk.RAISED,
            width=25,
        )
        self.add_model_btn.grid(row=3, column=0, padx=5, pady=5, sticky='we')

        self.fir_btn = tk.Button(
            self.master,
            text="Fit",
            command=self.fit,
            relief=tk.RAISED,
            width=25,
        )
        self.fir_btn.grid(row=2, column=1, padx=5, pady=5, sticky='s')

        # last occupied row in main frame
        self.current_row = 3
        self._plot_test = 1

    def fit(self):
        print('fit in ', self)

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

    def collect_params(self):
        params = {
            'k_weight': self.params_frame.kw_.get(),
            'k_shift': self.params_frame.k_shift_value.get(),
            'window': self.params_frame.wind.get(),
            'k_min':    self.params_frame.k_min.get(),
            'k_max':    self.params_frame.k_max.get(),
            'amp': self.params_frame.s02.get(),
        }
        print(params)
        return params


if __name__ == "__main__":
    root = tk.Tk()
    api = XAFSModelMixerAPI(root)
    root.mainloop()
