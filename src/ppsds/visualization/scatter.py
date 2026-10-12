import pickle

from .bounds import clamp_index, clamp_time_indexes


def plot_target(file_traj, k, time_indexes, ax, marker_color, zorder):
    with open(file_traj, 'rb') as inpt:
        traj = pickle.load(inpt)

    # Clamp: an out-of-range value warns instead of raising IndexError. The data was
    # already computed, so one figure should not fail the whole experiment.
    k = clamp_index(k, traj['x'].shape[0], what='trajectory index in plot_target')
    time_indexes = clamp_time_indexes(time_indexes, traj['x'].shape[1],
                                      what='time index in plot_target')

    target_plot = traj['x'][k, :]
    p3 = ax.scatter(target_plot[time_indexes, 0], target_plot[time_indexes, 1], color=marker_color, marker='*', zorder=zorder, s=15**2)

    return p3