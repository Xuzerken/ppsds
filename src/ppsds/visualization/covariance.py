import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse, Rectangle

import pickle
import os
from .bounds import clamp_index, clamp_len, clamp_time_indexes
from .curve import plot_traj
from .scatter import plot_target
from ppsds.numerics.linalg import eig_real


def plot_ellipses(file_predictor, k, time_indexes, ax, fig, color, zorder):
    with open(file_predictor, 'rb') as inpt:
        data = pickle.load(inpt)

    # Clamp: an out-of-range value warns instead of raising IndexError. The data was
    # already computed, so one figure should not fail the whole experiment.
    n_traj, n_steps = data['x_pred'].shape[0], data['x_pred'].shape[1]
    k = clamp_index(k, n_traj, what='trajectory index in plot_ellipses')
    time_indexes = clamp_time_indexes(time_indexes, n_steps,
                                      what='time index in plot_ellipses')

    mean_plot = data['x_pred'][k, :]
    cov_plot = data['Sigma_pred'][k, :]

    for t in time_indexes:
        mean = mean_plot[t]
        cov = cov_plot[t]
        eigenvalues, eigenvectors = eig_real(cov)
        theta = np.linspace(0, 2*np.pi, 1000)
        ellipses = np.sqrt(4.60517) * eigenvectors @ np.diag(np.sqrt(eigenvalues)) @ [np.sin(theta), np.cos(theta)]
        ax.plot(mean[0]+ellipses[0,:], mean[1]+ellipses[1,:], linewidth=2, alpha=1, color=color)
        # ax.plot(mean[0]+2*ellipses[0,:], mean[1]+2*ellipses[1,:], linewidth=2, alpha=0.6, color=color)
        # ax.plot(mean[0]+3*ellipses[0,:], mean[1]+3*ellipses[1,:], linewidth=2, alpha=0.3, color=color)

def plot_ellipses_compare_predictors(
        file_traj, 
        file_scores, 
        time_indexes, 
        k = 53, 
        xlim=None, 
        ylim=None, 
        colors = {
            'traj': '#8A7355',
            'Nominal':'#699ECA', 'Eig-DRPP':'#ED8828', 'Optimal':'#A8D3A0', 
            'Noise-DRPP': '#8D73BA'
        },
        local_ax=None
    ):
    fig, ax = plt.subplots()
    fig.set_size_inches(8,6)

    p_traj = plot_traj(file_traj=file_traj, k=k, traj_len=35, ax=ax, traj_color=colors['traj'], zorder=1)
    plot_ellipses(file_predictor=file_scores['Nominal'], k=k, time_indexes=time_indexes, ax=ax, fig=fig, color=colors['Nominal'], zorder=2)
    plot_ellipses(file_predictor=file_scores['Eig-DRPP'], k=k, time_indexes=time_indexes, ax=ax, fig=fig, color=colors['Eig-DRPP'], zorder=3)
    plot_ellipses(file_predictor=file_scores['Noise-DRPP'], k=k, time_indexes=time_indexes, ax=ax, fig=fig, color=colors['Noise-DRPP'], zorder=3)
    plot_ellipses(file_predictor=file_scores['Optimal'], k=k, time_indexes=time_indexes, ax=ax, fig=fig, color=colors['Optimal'], zorder=4)
    p_target = plot_target(file_traj=file_traj, k=k, time_indexes=time_indexes, ax=ax, marker_color='brown', zorder=6)

    ax.grid()
    ax.grid(which='major', color='#DDDDDD', linewidth=0.8)
    ax.set_xlabel('$x_1$',fontsize=18)
    ax.set_ylabel('$x_2$',fontsize=18)
    ax.xaxis.set_tick_params(labelsize=18)
    ax.yaxis.set_tick_params(labelsize=18)
    if xlim is not None:
        ax.set_xlim(xlim[0], xlim[1])
    if ylim is not None:
        ax.set_ylim(ylim[0], ylim[1])

    if local_ax is not None:
        x1, x2, y1, y2 = local_ax['x1'], local_ax['x2'], local_ax['y1'], local_ax['y2']
        axins = ax.inset_axes(
            local_ax['ax'],
            xlim=(x1, x2), ylim=(y1, y2))
        
        plot_traj(file_traj=file_traj, k=k, traj_len=35, ax=axins, traj_color=colors['traj'], zorder=1)
        plot_ellipses(file_predictor=file_scores['Nominal'], k=k, time_indexes=time_indexes, ax=axins, fig=fig, color=colors['Nominal'], zorder=2)
        plot_ellipses(file_predictor=file_scores['Eig-DRPP'], k=k, time_indexes=time_indexes, ax=axins, fig=fig, color=colors['Eig-DRPP'], zorder=3)
        plot_ellipses(file_predictor=file_scores['Noise-DRPP'], k=k, time_indexes=time_indexes, ax=axins, fig=fig, color=colors['Noise-DRPP'], zorder=3)
        plot_ellipses(file_predictor=file_scores['Optimal'], k=k, time_indexes=time_indexes, ax=axins, fig=fig, color=colors['Optimal'], zorder=4)
        p_target = plot_target(file_traj=file_traj, k=k, time_indexes=time_indexes, ax=axins, marker_color='brown', zorder=6)

        axins.grid()
        axins.grid(which='major', color='#DDDDDD', linewidth=0.8)
        axins.xaxis.set_tick_params(labelsize=16)
        axins.yaxis.set_tick_params(labelsize=16)
        ax.indicate_inset_zoom(axins, edgecolor="black", linestyle='--')

    extra = Rectangle((0, 0), 0.1, 0.1, fc="w", fill=False, edgecolor='none', linewidth=0)
    ellipses_eig_drpp = Ellipse((0, 0), 0.1, 0.1, fc=colors['Eig-DRPP'], fill=False, edgecolor=colors['Eig-DRPP'], linewidth=1)
    ellipses_noise_drpp = Ellipse((0, 0), 0.1, 0.1, fc=colors['Noise-DRPP'], fill=False, edgecolor=colors['Noise-DRPP'], linewidth=1)
    ellipses_nominal = Ellipse((0, 0), 0.1, 0.1, fc=colors['Nominal'], fill=False, edgecolor=colors['Nominal'], linewidth=1)
    ellipses_optimal = Ellipse((0, 0), 0.1, 0.1, fc=colors['Optimal'], fill=False, edgecolor=colors['Optimal'], linewidth=1)

    legend_handle = [extra]*2 + [extra, ellipses_nominal] + [extra, ellipses_noise_drpp] + [extra, ellipses_eig_drpp] + [extra, ellipses_optimal] 
    legend_label = ['', 'predictive 90% CR', 'Nominal', '', 'Noise-DRPP', '', 'Eig-DRPP', '', 'Optimal', '', ]
    legend_1 = plt.legend(legend_handle, legend_label, loc='lower right', ncol=5, fontsize=12, handletextpad = -2.3)
    legend_2 = plt.legend([p_traj, p_target], ['state trajectory', 'prediction target'], loc='upper left', fontsize=12)
    ax.add_artist(legend_1)
    ax.add_artist(legend_2)

    fig.savefig(os.path.join(os.path.dirname(file_traj), 'traj_ellipses.pdf'), bbox_inches='tight')


def plot_ellipses_over_traj(file_data, k, traj_len, time_indexes, ax, fig, color, marker_color, traj_color):
    with open(file_data, 'rb') as inpt:
        data = pickle.load(inpt)

    # Clamp (same as plot_ellipses)
    n_traj, n_steps = data['x'].shape[0], data['x'].shape[1]
    k = clamp_index(k, n_traj, what='trajectory index in plot_ellipses_over_traj')
    traj_len = clamp_len(traj_len, n_steps, what='traj_len in plot_ellipses_over_traj')
    time_indexes = clamp_time_indexes(time_indexes, traj_len,
                                      what='time index in plot_ellipses_over_traj')

    traj_plot = data['x'][k, :traj_len]
    mean_plot = data['x_pred'][k, :traj_len]
    cov_plot = data['Sigma_pred'][k, :traj_len]

    p1, = ax.plot(traj_plot[:,0], traj_plot[:,1], zorder=1, color=traj_color, alpha=0.4)

    for t in time_indexes:
        mean = mean_plot[t]
        cov = cov_plot[t]
        eigenvalues, eigenvectors = eig_real(cov)
        theta = np.linspace(0, 2*np.pi, 1000)
        ellipses = np.sqrt(4.60517) * eigenvectors @ np.diag(np.sqrt(eigenvalues)) @ [np.sin(theta), np.cos(theta)]
        p2 = ax.plot(mean[0]+ellipses[0,:], mean[1]+ellipses[1,:], linewidth=2, alpha=1, color=color)
    
    p3 = ax.scatter(traj_plot[time_indexes, 0], traj_plot[time_indexes, 1], color=marker_color, marker='*', zorder=3, s=15**2)

    return [p1, p2, p3]

def plot_ellipses_compare_predictors_worst_case(
        file_scores, 
        time_indexes, 
        k=23, 
        colors = {
            'traj': '#848484',
            'Nominal':'#699ECA', 'Eig-DRPP':'#ED8828', 'Optimal':'#A8D3A0', 
            'Noise-DRPP': '#8D73BA'
        },
        xlim=None, 
        ylim=None
    ):
    fig, ax = plt.subplots()
    fig.set_size_inches(8,6)
    p_eig_drpp = plot_ellipses_over_traj(file_data=file_scores['Eig-DRPP'], k=k, traj_len=35, time_indexes=time_indexes, ax=ax, fig=fig, color=colors['Eig-DRPP'], marker_color=colors['Eig-DRPP'], traj_color=colors['Eig-DRPP'])
    p_noise_drpp = plot_ellipses_over_traj(file_data=file_scores['Noise-DRPP'], k=k, traj_len=35, time_indexes=time_indexes, ax=ax, fig=fig, color=colors['Noise-DRPP'], marker_color=colors['Noise-DRPP'], traj_color=colors['Noise-DRPP'])
    p_nominal = plot_ellipses_over_traj(file_data=file_scores['Nominal'], k=k, traj_len=35, time_indexes=time_indexes, ax=ax, fig=fig, color=colors['Nominal'], marker_color=colors['Nominal'], traj_color=colors['Nominal'])

    ax.grid()
    ax.grid(which='major', color='#DDDDDD', linewidth=0.8)
    ax.set_xlabel('$x_1$',fontsize=18)
    ax.set_ylabel('$x_2$',fontsize=18)
    ax.xaxis.set_tick_params(labelsize=18)
    ax.yaxis.set_tick_params(labelsize=18)
    if xlim is not None:
        ax.set_xlim(xlim[0], xlim[1])
    if ylim is not None:
        ax.set_ylim(ylim[0], ylim[1])

    extra = Rectangle((0, 0), 0.1, 0.1, fc="w", fill=False, edgecolor='none', linewidth=0)
    ellipses_eig_drpp = Ellipse((0, 0), 0.1, 0.1, fc=colors['Eig-DRPP'], fill=False, edgecolor=colors['Eig-DRPP'], linewidth=1)
    ellipses_noise_drpp = Ellipse((0, 0), 0.1, 0.1, fc=colors['Noise-DRPP'], fill=False, edgecolor=colors['Noise-DRPP'], linewidth=1)
    ellipses_nominal = Ellipse((0, 0), 1, 1, fc=colors['Nominal'], fill=False, edgecolor=colors['Nominal'], linewidth=1)

    legend_handle = [extra]*5 + [p_nominal[0], p_nominal[2], ellipses_nominal] + [extra] + [p_noise_drpp[0], p_noise_drpp[2], ellipses_noise_drpp] + [extra] + [p_eig_drpp[0], p_eig_drpp[2], ellipses_eig_drpp]
    legend_label = ['', 'state trajectory', 'prediction target', 'predictive 90% CR', 'Nominal', '', '', '','Noise-DRPP', '','','','Eig-DRPP', '','','']
    ax.legend(legend_handle, legend_label, loc=3, ncol=4, fontsize=12, handletextpad = -2.3)

    fig.savefig(os.path.join(os.path.dirname(file_scores['Nominal']), 'traj_ellipse.pdf'), bbox_inches='tight')