import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from scipy.stats import multivariate_normal
from matplotlib.patches import Rectangle, Ellipse
from .bounds import clamp_index, clamp_len, clamp_time_indexes
from .scatter import plot_target
from .curve import plot_traj
colors = ['#4077B1', '#BDB8D9', '#9DCEC8', '#EA8575' , '#D12627']

def plot_contours(file_predictor, k, time_indexes, ax, fig, contour_width, cmap_color, zorder, alpha, cax):
    with open(file_predictor, 'rb') as inpt:
        data = pickle.load(inpt)

    # Clamp: an out-of-range value warns instead of raising IndexError.
    k = clamp_index(k, data['x_pred'].shape[0], what='trajectory index in plot_contours')
    time_indexes = clamp_time_indexes(time_indexes, data['x_pred'].shape[1],
                                      what='time index in plot_contours')

    mean_plot = data['x_pred'][k, :]
    cov_plot = data['Sigma_pred'][k, :]

    def f(x, y, mean, cov):
        return multivariate_normal(mean, cov).pdf([x, y])
    vmax = np.max([f(mean_plot[t][0], mean_plot[t][1], mean_plot[t], cov_plot[t]) for t in time_indexes])
    vmin = np.min([f(mean_plot[t][0]-contour_width, mean_plot[t][1]-contour_width, mean_plot[t], cov_plot[t]) for t in time_indexes])

    for t in time_indexes:
        mean = mean_plot[t]
        cov = cov_plot[t]
        x = np.linspace(mean[0]-contour_width, mean[0]+contour_width, contour_width * 5)
        y = np.linspace(mean[1]-contour_width, mean[1]+contour_width, contour_width * 5)
        X, Y = np.meshgrid(x, y)
        z = np.array([f(x, y, mean, cov) for x,y in zip(np.ravel(X), np.ravel(Y))])
        Z = z.reshape(X.shape)

        p2 = ax.contour(X, Y, Z, cmap=cmap_color, zorder=zorder, levels=np.linspace(vmin, vmax, 50), vmax=vmax, vmin=vmin, alpha=alpha)
    
    cb = fig.colorbar(p2, cax=ax.inset_axes(cax))
    cb.set_ticks([float(f'{v:.3f}') for v in np.linspace(vmin+0.001, vmax-0.0005, 6)])

    return [p2, cb]

def plot_contours_compare_predictors(file_traj, file_nominal, file_optimal, file_drpp, time_indexes, k = 53):
    fig, ax = plt.subplots()
    fig.set_size_inches(9,6)

    p_traj = plot_traj(file_traj=file_traj, k=k, traj_len=35, ax=ax, traj_color='slategrey', zorder=1)
    # The three contour panels are drawn for their side effect; only the trajectory and target
    # handles are needed for the legend at the bottom of this function.
    plot_contours(file_predictor=file_nominal, k=k, time_indexes=time_indexes, ax=ax, fig=fig, contour_width=12, cmap_color='Blues', zorder=4, alpha=1, cax=[1.22, 0.1, 0.02, 0.8])
    plot_contours(file_predictor=file_optimal, k=k, time_indexes=time_indexes, ax=ax, fig=fig, contour_width=12, cmap_color='Greens', zorder=3, alpha=1, cax=[1.12, 0.1, 0.02, 0.8])
    plot_contours(file_predictor=file_drpp, k=k, time_indexes=time_indexes, ax=ax, fig=fig, contour_width=12, cmap_color='Oranges', zorder=2, alpha=1, cax=[1.02, 0.1, 0.02, 0.8])
    p_target = plot_target(file_traj=file_traj, k=k, time_indexes=time_indexes, ax=ax, marker_color='brown', zorder=6)

    ax.grid()
    ax.set_xlabel('$x_1$',fontsize=18)
    ax.set_ylabel('$x_2$',fontsize=18)
    ax.xaxis.set_tick_params(labelsize=18)
    ax.yaxis.set_tick_params(labelsize=18)

    extra = Rectangle((0, 0), 0.1, 0.1, fc="w", fill=False, edgecolor='none', linewidth=0)
    contour_drpp = Ellipse((0, 0), 0.1, 0.1, fc='darkorange', fill=True, edgecolor='darkorange', linewidth=1)
    contour_nominal = Ellipse((0, 0), 0.1, 0.1, fc='tab:blue', fill=True, edgecolor='tab:blue', linewidth=1)
    contour_optimal = Ellipse((0, 0), 0.1, 0.1, fc='tab:green', fill=True, edgecolor='tab:green', linewidth=1)

    legend_handle = [extra]*3 + [contour_drpp, extra] + [contour_optimal, extra] + [contour_nominal]
    legend_label = ['', 'predictive pdf', 'DRPP', '', 'optimal', '', 'nominal', '',]
    legend_1 = plt.legend(legend_handle, legend_label, loc=1, ncol=4, fontsize=12, handletextpad = -2.3)
    legend_2 = plt.legend([p_traj, p_target], ['state trajectory', 'prediction target'], loc=3, fontsize=12)
    ax.add_artist(legend_1)
    ax.add_artist(legend_2)

    fig.savefig(os.path.join(os.path.dirname(file_traj), 'traj.pdf'), bbox_inches='tight')


def plot_contour_over_traj(file_data, k, traj_len, contour_width, time_indexes, ax, fig, cmap_color, marker_color, traj_color, cax):
    with open(file_data, 'rb') as inpt:
        data = pickle.load(inpt)

    # Clamp (same as plot_contours)
    k = clamp_index(k, data['x'].shape[0], what='trajectory index in plot_contour_over_traj')
    traj_len = clamp_len(traj_len, data['x'].shape[1],
                         what='traj_len in plot_contour_over_traj')
    time_indexes = clamp_time_indexes(time_indexes, traj_len,
                                      what='time index in plot_contour_over_traj')

    traj_plot = data['x'][k, :traj_len]
    mean_plot = data['x_pred'][k, :traj_len]
    cov_plot = data['Sigma_pred'][k, :traj_len]

    p1, = ax.plot(traj_plot[:,0], traj_plot[:,1], zorder=1, color=traj_color)

    def f(x, y, mean, cov):
        return multivariate_normal(mean, cov).pdf([x, y])
    vmax = np.max([f(mean_plot[t][0], mean_plot[t][1], mean_plot[t], cov_plot[t]) for t in time_indexes])
    vmin = np.min([f(mean_plot[t][0]-contour_width, mean_plot[t][1]-contour_width, mean_plot[t], cov_plot[t]) for t in time_indexes])

    for t in time_indexes:
        mean = mean_plot[t]
        cov = cov_plot[t]
        x = np.linspace(mean[0]-contour_width, mean[0]+contour_width, contour_width * 5)
        y = np.linspace(mean[1]-contour_width, mean[1]+contour_width, contour_width * 5)
        X, Y = np.meshgrid(x, y)
        z = np.array([f(x, y, mean, cov) for x,y in zip(np.ravel(X), np.ravel(Y))])
        Z = z.reshape(X.shape)

        p2 = ax.contour(X, Y, Z, cmap=cmap_color, zorder=2, levels=np.linspace(vmin, vmax, 50), vmax=vmax, vmin=vmin)
    
    cb = fig.colorbar(p2, cax=ax.inset_axes(cax))
    cb.set_ticks([float(f'{v:.3f}') for v in np.linspace(vmin+0.001, vmax-0.001, 6)])
    p3 = ax.scatter(traj_plot[time_indexes, 0], traj_plot[time_indexes, 1], color=marker_color, marker='*', zorder=3)

    return [p1, [p2, cb], p3]

def plot_compare_nominal_and_drpp_on_one_trajectory(file_data_nominal, file_data_drpp, time_indexes):
    fig, ax = plt.subplots()
    fig.set_size_inches(8,6)
    p_drpp = plot_contour_over_traj(file_data=file_data_drpp, k=21, traj_len=35, contour_width=12, time_indexes=time_indexes, ax=ax, fig=fig, cmap_color='Oranges', marker_color='tab:red', traj_color='lightcoral', cax=[1.12, 0.1, 0.02, 0.8])
    p_nominal = plot_contour_over_traj(file_data=file_data_nominal, k=56, traj_len=35, contour_width=12, time_indexes=time_indexes, ax=ax, fig=fig, cmap_color='Blues', marker_color='tab:blue', traj_color='lightsteelblue', cax=[1.01, 0.1, 0.02, 0.8])
    ax.grid()
    ax.set_xlabel('$x_1$',fontsize=18)
    ax.set_ylabel('$x_2$',fontsize=18)
    ax.xaxis.set_tick_params(labelsize=18)
    ax.yaxis.set_tick_params(labelsize=18)

    extra = Rectangle((0, 0), 0.1, 0.1, fc="w", fill=False, edgecolor='none', linewidth=0)
    contour_drpp = Ellipse((0, 0), 0.1, 0.1, fc='darkorange', fill=True, edgecolor='darkorange', linewidth=1)
    contour_nominal = Ellipse((0, 0), 1, 1, fc='tab:blue', fill=True, edgecolor='tab:blue', linewidth=1)

    legend_handle = [extra]*5 + [p_nominal[0], p_nominal[2], contour_nominal] + [extra] + [p_drpp[0], p_drpp[2], contour_drpp]
    legend_label = ['', 'state trajectory', 'prediction target', 'predictive pdf', 'nominal', '', '', '','DRPP', '','','']
    ax.legend(legend_handle, legend_label, loc=3, ncol=3, fontsize=12, handletextpad = -2.3)

    fig.savefig(os.path.join(os.path.dirname(file_data_nominal), 'traj.pdf'), bbox_inches='tight')