import os
import tkinter as tk
from tkinter.filedialog import askopenfilename
from tkinter.simpledialog import askstring, askfloat, askinteger
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from utils.dataset import Dataset, DataManager


class ParamsFrame(tk.Frame):
    def __init__(self, parent):
        super().__init__(parent, border=2, relief=tk.GROOVE)

        tk.Label(self, text="k-shift: ").grid(row=0, column=0, padx=5, pady=5, sticky='e')
        self.k_shift_value = tk.DoubleVar(value=0.0)
        # TODO: check if entry can by created without assigning if value can be obtained by tkVar()
        self.k_shift = tk.Entry(self, width=15, textvariable=self.k_shift_value)
        self.k_shift.grid(row=0, column=1, padx=5, pady=5)

        # first row
        tk.Label(self, text="Amp.: ").grid(row=0, column=2, padx=5, pady=5, sticky='e')
        self.s02 = tk.DoubleVar(value=0.81)
        self.amp = tk.Entry(self, width=15, textvariable=self.s02)
        self.amp.grid(row=0, column=3, padx=5, pady=5)

        # Create window function selector
        tk.Label(self, text="Window function:").grid(row=0, column=4, padx=5, pady=5)
        self.wind = tk.StringVar(self)
        choices = ('kaiser', 'bartlett', 'blackman', 'hamming', 'hanning')
        self.wind.set(choices[0])  # set the default option
        self.wind_func_menu = tk.OptionMenu(self, self.wind, *choices)
        self.wind_func_menu.grid(row=0, column=5, padx=5, pady=5, sticky="ew")
        self.wind_func_menu.configure(width=10)

        # second row:
        tk.Label(self, text="k min: ").grid(row=1, column=0, padx=5, pady=5, sticky='e')
        self.k_min = tk.DoubleVar(value=3.5)
        self.kmin = tk.Entry(self, width=15, textvariable=self.k_min)
        self.kmin.grid(row=1, column=1, padx=5, pady=5)

        tk.Label(self, text="k max: ").grid(row=1, column=2, padx=5, pady=5, sticky='e')
        self.k_max = tk.DoubleVar(value=12.0)
        self.kmax = tk.Entry(self, width=15, textvariable=self.k_max)
        self.kmax.grid(row=1, column=3, padx=5, pady=5)

        tk.Label(self, text="k-weight: ").grid(row=1, column=4, padx=5, pady=5, sticky='e')
        self.kw_ = tk.IntVar(value=1)
        self.kw = tk.Entry(self, width=15, textvariable=self.kw_)
        self.kw.grid(row=1, column=5, padx=5, pady=5)


class ExperimentalDataFrame(tk.Frame):
    def __init__(self, parent, gui):
        super().__init__(parent, border=2, relief=tk.GROOVE)
        self.gui = gui
        self.dataset: Dataset = None
        self.add_exp_data = tk.Button(
            self,
            text="Import Experimental data",
            relief=tk.RAISED,
            command=lambda: self.gui.import_file(self),
            width=25,
        )
        self.add_exp_data.grid(row=0, column=0, padx=5, pady=5, stick='ws')

        self.name = tk.Entry(self, width=30, text='No imported data', disabledbackground='white',
                             disabledforeground='black')
        self.name.grid(row=1, column=0, padx=5, pady=5, stick='ws')
        self.name.config(state=tk.DISABLED)

        self.change_name = tk.Button(
            self,
            text="Rename",
            relief=tk.RAISED,
            command=lambda: self.gui.rename_dataset(self),
            width=10,
        )
        self.change_name.grid(row=1, column=1, padx=5, pady=5, stick='ws')
        self.name.bind('<Double-Button-1>', self.double_click_handler)

    def double_click_handler(self, event):
        self.gui.rename_dataset(self)


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


class ModelDataFrame(tk.Frame):
    # TODO: inherit from experimental Frame and do review of the class
    def __init__(self, parent, gui, dataset):

        super().__init__(parent, border=2, relief=tk.GROOVE)
        self.gui = gui
        self.dataset: Dataset = dataset

        self.name = tk.Entry(self, width=30, disabledbackground='white', disabledforeground='black')
        self.name.grid(row=0, column=0, padx=5, pady=5, stick='ws')
        self.name.insert(0, dataset.name)
        self.name.config(state=tk.DISABLED)

        self.name.bind('<Double-Button-1>', self.double_click_handler)

        # TODO: remove the 'rename' button
        tk.Label(self, text="weight: ").grid(row=0, column=1, padx=5, pady=5, sticky='e')
        self.w = tk.DoubleVar(value=0.0)
        self.weight = tk.Entry(self, width=15, textvariable=self.w)
        self.weight.grid(row=0, column=2, padx=5, pady=5, stick='ws')

        self.hold_w_value = tk.BooleanVar()
        self.hold_w = tk.Checkbutton(self, text="Freeze weight", variable=self.hold_w_value)
        self.hold_w.grid(row=0, column=3, stick='sn')

        self.change_name = tk.Button(
            self,
            text="Rename",
            relief=tk.RAISED,
            command=lambda: self.gui.rename_dataset(self),
            width=25,
        )
        self.change_name.grid(row=1, column=0, padx=5, pady=5, stick='ws')

        self.reopen = tk.Button(
            self,
            text="...",
            relief=tk.RAISED,
            command=lambda: self.gui.import_file(self),
            width=10,
        )
        self.reopen.grid(row=1, column=1, padx=5, pady=5, stick='ws')

        self.remove = tk.Button(
            self,
            text="X",
            relief=tk.RAISED,
            command=lambda: self.gui.delete_data_frame(self),
            width=10,
        )
        self.remove.grid(row=1, column=3, padx=5, pady=5, stick='e')
        # TODO: add method that collect al parameters for given dataset from corresponding frame

    def double_click_handler(self, event):
        self.gui.rename_dataset(self)

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
        self.experimental_data = ExperimentalDataFrame(self.master, gui=self)
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
        filepath = askopenfilename(
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
            invoker.name.config(state=tk.NORMAL)
            invoker.name.delete(0, tk.END)
            invoker.name.insert(0, os.path.splitext(os.path.basename(filepath))[0])
            invoker.name.config(state=tk.DISABLED)

    def rename_dataset(self, invoker):
        if invoker.dataset:
            current_name = invoker.name.get()
            new_name = askstring('Change data label', 'Change name', initialvalue=current_name)
            if new_name and new_name != current_name:
                ds_idx = self.data.index(invoker.dataset)
                invoker.name.config(state=tk.NORMAL)
                invoker.name.delete(0, tk.END)
                invoker.name.insert(0, new_name)
                invoker.name.config(state=tk.DISABLED)
                invoker.dataset.name = new_name
                self.data[ds_idx] = invoker.dataset

    def add_model(self):
        filepath = askopenfilename(
            filetypes=(
                ("data", "*.dat"),
                ("text", "*.txt"),
                ("All files", "*.*")
            )
        )
        if filepath:
            dataset = Dataset(filepath)
            self.data.append(dataset)
            new_data_frame = ModelDataFrame(self.master, gui=self, dataset=dataset)
            new_data_frame.grid(row=self.current_row+1, column=0, padx=5, pady=5, sticky='we')
            self.current_row += 1

    def collect_params(self):
        params = {
            'amp': float(self.params_frame.s02.get()),
            'k_weight': float(self.params_frame.kw_.get()),
            'k_min': float(self.params_frame.kmin.get()),
            'k_max': float(self.params_frame.kmax.get()),
            'window': self.params_frame.wind.get(),
            'k_shift': float(self.params_frame.k_shift_value.get()),
        }
        return params


if __name__ == "__main__":
    root = tk.Tk()
    api = XAFSModelMixerAPI(root)
    root.mainloop()
