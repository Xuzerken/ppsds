import numpy as np
import matplotlib.pyplot as plt
import pickle
import os

from .bounds import clamp_horizon, clamp_index, clamp_len
from .data_transform import cum_average
from .graph_format import set_figure_formats


def plot_error_bar(
        file_scores, 
        upper,
        lower,
        ylim=None,  
        markers = {'Nominal':'o', 'Eig-DRPP':'d', 'Optimal':'v', 'Noise-DRPP': 's'}, 
        colors = {
            'Nominal':'#699ECA', 'Eig-DRPP':'#ED8828', 'Optimal':'#A8D3A0', 
            'Noise-DRPP': '#8D73BA'},
        capsizes = {'Nominal':8, 'Eig-DRPP':6, 'Noise-DRPP':6, 'Optimal':4},
        show_local=False):
    fig, ax = plt.subplots()
    fig.set_size_inches(8,6)
    for name, file_score in file_scores.items():
        with open(file_score, 'rb') as f:
            dct = pickle.load(f)
        T = clamp_horizon(31, dct['score'].shape[1] - 1,
                          what='time horizon in plot_error_bar')
        score = cum_average(dct['score'][:, 1:T+1])
        mean = np.mean(score, axis=0)
        ci = np.quantile(score, axis=0, q = [.05, 0.95])
        yerr = np.zeros((2,T))
        yerr[0] = mean - ci[0]
        yerr[1] = ci[1] - mean
        ax.errorbar(range(1,T+1), mean, yerr=yerr, label=name, capsize=capsizes[name], capthick=1, elinewidth=1, linewidth=2, marker=markers[name], color=colors[name], errorevery=(0,2))
    
    ax.axhline(upper, label='upper bound', linewidth=2, color=colors['Noise-DRPP'], linestyle='dotted')
    ax.axhline(lower, label='lower bound', linewidth=2, color=colors['Eig-DRPP'], linestyle='dotted')

    set_figure_formats(ax=ax)
    if ylim is not None:
        ax.set_ylim(ylim[0], ylim[1])
    ax.set_xticks(np.arange(1, T + 1, 5))

    if show_local:
        # The inset window originally spanned 16.5~31.5 (for T=31). Pull the right edge
        # in as the horizon shrinks, otherwise it shows a stretch of blank space.
        x1, x2, y1, y2 = 16.5, min(31.5, T + 0.5), -6.5, -3.5
        axins = ax.inset_axes(
            [0.4, 0.1, 0.57, 0.55],
            xlim=(x1, x2), ylim=(y1, y2))
        for name, file_score in file_scores.items():
            with open(file_score, 'rb') as f:
                dct = pickle.load(f)
            T = clamp_horizon(31, dct['score'].shape[1] - 1,
                              what='inset time horizon in plot_error_bar')
            score = cum_average(dct['score'][:, 1:T+1])
            mean = np.mean(score, axis=0)
            # ci = np.quantile(score, axis=0, q = [.025, 0.975])
            ci = np.quantile(score, axis=0, q = [.05, 0.95])
            yerr = np.zeros((2,T))
            yerr[0] = mean - ci[0]
            yerr[1] = ci[1] - mean
            axins.errorbar(range(1,T+1), mean, yerr=yerr, label=name, capsize=capsizes[name], capthick=1, elinewidth=1, linewidth=2, marker=markers[name], color=colors[name], errorevery=(0,2))
        
        axins.axhline(upper, label='upper bound', linewidth=2, color=colors['Noise-DRPP'], linestyle='dotted')
        axins.axhline(lower, label='lower bound', linewidth=2, color=colors['Eig-DRPP'], linestyle='dotted')
        axins.grid(which='major', color='#DDDDDD', linewidth=0.8)
        axins.grid(which='minor', color='#EEEEEE', linestyle=':', linewidth=0.5)
        axins.minorticks_on()
        axins.xaxis.set_tick_params(labelsize=16)
        axins.yaxis.set_tick_params(labelsize=16)

        # change the connection lines to an arrow
        # ax.indicate_inset_zoom(axins, edgecolor="black", linestyle='--') 
        ax.annotate('', xy=(23.5, -10.5), xytext=(24,-6.5), arrowprops=dict(facecolor='#595959', edgecolor='#595959', width=0.8, headwidth=6))

    # Locate the output directory from the file we **explicitly picked**. The loop
    # variable file_score was used before, which only held when every file happened to
    # share a directory and it was non-empty.
    _ref = next(iter(file_scores.values()))
    fig.savefig(os.path.join(os.path.dirname(_ref), 'error_bar.pdf'), bbox_inches='tight')

def plot_traj(file_traj, k, traj_len, ax, traj_color, zorder):
    with open(file_traj, 'rb') as inpt:
        traj = pickle.load(inpt)

    # Clamp: an out-of-range value warns instead of raising IndexError. The data was
    # already computed, so one figure should not fail the whole experiment.
    k = clamp_index(k, traj['x'].shape[0], what='trajectory index in plot_traj')
    traj_len = clamp_len(traj_len, traj['x'].shape[1], what='traj_len in plot_traj')

    traj_plot = traj['x'][k, :traj_len]
    p1, = ax.plot(traj_plot[:,0], traj_plot[:,1], zorder=zorder, color=traj_color, alpha=0.5)
    return p1

