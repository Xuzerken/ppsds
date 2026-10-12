import matplotlib.pyplot as plt

class Graph2DFormat():
    def __init__(self, width=8, height=6, xlabel='time step', ylabel='score', label_fontsize=18, legend_loc='lower center', legend_fontsize=14, xticks=None, yticks=None, tick_fontsize=18, xlim=None, ylim=None, show_grid=True):
        self.width = width
        self.height = height
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.label_fontsize = label_fontsize
        self.legend_loc = legend_loc
        self.legend_fontsize= legend_fontsize
        self.xticks = xticks
        self.yticks = yticks
        self.tick_fontsize = tick_fontsize
        self.xlim = xlim
        self.ylim = ylim
        self.show_grid = show_grid
        
    def set_formate(self, fig, ax):
        fig.set_size_inches(self.width, self.height)
        ax.set_xlabel(self.xlabel, fontsize=self.label_fontsize)
        ax.set_ylabel(self.ylabel, fontsize=self.label_fontsize)
        ax.legend(loc=self.legend_loc, fontsize=self.legend_fontsize)
        if self.xlim is not None:
            ax.set_xlim(self.xlim[0], self.xlim[1])
        if self.ylim is not None:
            ax.set_ylim(self.ylim[0], self.ylim[1])
        if self.xticks is not None:
            ax.set_xticks(self.xticks)
        if self.yticks is not None:
            ax.set_yticks(self.yticks)
        ax.xaxis.set_tick_params(labelsize=self.tick_fontsize)
        ax.yaxis.set_tick_params(labelsize=self.tick_fontsize)
        ax.grid(visible=self.show_grid)

class Graph2D():
    def __init__(self, data, file_save, formatter):
        fig, ax = plt.subplots()
        self.fig = fig
        self.ax = ax
        self.data = data
        self.file_save = file_save
        self.formatter = formatter

    def plot(self, set_fig=False, show_fig=False, save_fig=True):
        if set_fig:
            self.set_formate()
        if show_fig:
            self.fig.show()
        if save_fig:
            self.save()

    def save(self):
        self.fig.savefig(self.file_save, bbox_inches='tight')

    def set_formate(self):
        self.formatter.set_formate(self.fig, self.ax)

