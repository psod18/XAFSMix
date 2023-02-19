import os
import tkinter as tk
import numpy as np
from tkinter import (
    filedialog,
    simpledialog,
)
from utils.custom_frames import (
    PlotWindow,
    ParamsFrame,
    ExperimentalDataFrame,
    ModelDataFrame,
    FittingFrame,
)
from utils.dataset import Dataset
from utils.tools import calculate_fft


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

        self.k_space_fit_plot = PlotWindow(self, 'k')
        self.r_space_fit_plot = PlotWindow(self, 'r')

        # Frames for global parameter (k-shift, s02, etc.) and plot buttons (k- and R-space
        self.params_frame = ParamsFrame(parent=self.master)
        self.params_frame.grid(row=0, rowspan=2, column=0, padx=10, pady=10, sticky='we')

        # Plot buttons on main frame
        self.plot_k_btn = tk.Button(
            self.master,
            text="Plot k-space",
            command=lambda: self.k_space_plot.update_plot(self.get_data_for_plot('k')),
            relief=tk.RAISED,
            width=25,
        )
        self.plot_k_btn.grid(row=0, column=1, columnspan=3, padx=5, pady=5)

        self.x_min_k = tk.DoubleVar(value=2.5)
        self.xmin_k = tk.Entry(self.master, width=5, textvariable=self.x_min_k, state=tk.DISABLED,
                               disabledbackground='white', disabledforeground='black')
        self.xmin_k.grid(row=1, column=1, padx=5, pady=5, sticky='e')
        self.xmin_k.bind('<Double-Button-1>', self.set_axis_range)

        tk.Label(self.master, text=': min - max :').grid(row=1, column=2, padx=5, pady=5, sticky='we')
        self.x_max_k = tk.DoubleVar(value=16.0)
        self.xmax_k = tk.Entry(self.master, width=5, textvariable=self.x_max_k, state=tk.DISABLED,
                               disabledbackground='white', disabledforeground='black')
        self.xmax_k.grid(row=1, column=3, padx=5, pady=5, sticky='w')
        self.xmax_k.bind('<Double-Button-1>', self.set_axis_range)

        self.plot_R_btn = tk.Button(
            self.master,
            text="Plot R-space",
            command=lambda: self.r_space_plot.update_plot(self.get_data_for_plot('r')),
            relief=tk.RAISED,
            width=25,
        )
        self.plot_R_btn.grid(row=2, column=1, columnspan=3, padx=5, pady=5)

        self.x_min_r = tk.DoubleVar(value=1.0)
        self.xmin_r = tk.Entry(self.master, width=5, textvariable=self.x_min_r, state=tk.DISABLED,
                               disabledbackground='white', disabledforeground='black')
        self.xmin_r.grid(row=3, column=1, padx=5, pady=5, sticky='e')
        self.xmin_r.bind('<Double-Button-1>', self.set_axis_range)

        tk.Label(self.master, text=': min - max :').grid(row=3, column=2, padx=5, pady=5, sticky='we')
        self.x_max_r = tk.DoubleVar(value=6.0)
        self.xmax_r = tk.Entry(self.master, width=5, textvariable=self.x_max_r, state=tk.DISABLED,
                               disabledbackground='white', disabledforeground='black')
        self.xmax_r.grid(row=3, column=3, padx=5, pady=5, sticky='w')
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
        self.add_model_btn.grid(row=4, column=0, rowspan=2, padx=5, pady=5, sticky='we')

        self.fit_frame = FittingFrame(gui=self)
        self.fit_frame.grid(row=4, rowspan=2, column=1, columnspan=3, padx=5, pady=5, sticky='enw')

        # last occupied row in main frame
        self.current_row = 5

    def set_axis_range(self, event):
        val = simpledialog.askfloat(title="Set axis range", prompt="Set axis value limit")
        if val is not None:
            event.widget.configure(state=tk.NORMAL)
            event.widget.delete(0, tk.END)
            event.widget.insert(0, val)
            event.widget.configure(state=tk.DISABLED)

    def delete_data_frame(self, frame: ModelDataFrame):
        self.data.remove(frame.dataset)
        frame.grid_forget()
        frame.destroy()
        # self.current_row -= 1

    def import_file(self, invoker):
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
            if invoker.dataset is not None:
                idx = self.data.index(invoker.dataset)
                # self.data.remove(invoker.dataset)
                dataset.lw = invoker.line_width.get()
                dataset.color = invoker.color.cget('bg')
                invoker.dataset.is_experimental = invoker.experimental
                self.data[idx] = dataset
            else:
                self.data.append(dataset)
            invoker.dataset = dataset
            invoker.dataset.is_experimental = invoker.experimental
            invoker.line_width.set(dataset.lw)
            if invoker.line_style_menu['state'] == 'disabled':
                invoker.line_style_menu.config(state='normal')
            else:
                invoker.dataset.ls = invoker.line_style.get()
            invoker.name_value.set(os.path.splitext(os.path.basename(filepath))[0])
            invoker.k_shift.set(.0)

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
            new_data_frame.k_shift.set(dataset.k_shift)
            new_data_frame.line_style_menu.config(state='normal')
            self.current_row += 1

    def window_function(self, k):
        k_wind = np.zeros(len(k))
        dx = 1

        # get windows target area from GUI widget
        kmin = self.params_frame.k_min.get()
        kmax = self.params_frame.k_max.get()
        window = self.params_frame.wind.get()

        # get indices of k-values for allocation window function borders
        kmin_ind = np.where(k < kmin)[0][-1]
        kmin_ind1 = np.where(k < (kmin + dx))[0][-1]
        kmax_ind = np.where(k > kmax)[0][0]
        kmax_ind1 = np.where(k > (kmax - dx))[0][0]

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

        win_shift = len(dx1) // 2

        k_wind[kmin_ind - win_shift:kmin_ind1 - win_shift + 1] = dx1
        k_wind[kmax_ind1 + win_shift:kmax_ind + win_shift + 1] = dx2
        k_wind[kmin_ind1 - win_shift:kmax_ind1 + win_shift] = max(init_window)
        return k_wind

    def get_data_for_plot(self, space):
        out = []
        # Get necessary params from API entry fields
        # k_shift = self.params_frame.k_shift_value.get()
        s02 = self.params_frame.s02.get()
        kw = self.params_frame.kw_.get()

        if space == 'k':
            # TODO: add possibility to plot windowed data
            for dataset in self.data:
                attr_for_plot = {'label': dataset.name, 'ls': dataset.ls, 'lw': dataset.lw, 'c': dataset.color}
                if dataset.is_experimental:
                    k, chi = dataset.get_k_chi(kw=kw, s02=1.)
                else:
                    k, chi = dataset.get_k_chi(kw=kw, s02=s02)
                out.append([k, chi, attr_for_plot])

        elif space == 'r':
            for dataset in self.data:
                attr_for_plot = {'label': dataset.name, 'ls': dataset.ls, 'lw': dataset.lw, 'c': dataset.color}
                if dataset.is_experimental:
                    k, chi = dataset.get_k_chi(kw=kw, s02=1.)
                else:
                    k, chi = dataset.get_k_chi(kw=kw, s02=s02)
                window = self.window_function(k)
                chi = chi * window
                r, ft = calculate_fft(chi=chi, k_step=0.05)
                out.append([r, ft, attr_for_plot])
        else:
            print(f"Unknown space {space}")

        return out


if __name__ == "__main__":
    root = tk.Tk()
    api = XAFSModelMixerAPI(root)
    root.mainloop()
