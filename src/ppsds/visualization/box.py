import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
from .graph_format import set_figure_formats
from .data_transform import cum_average, data_to_dict
colors = ['#4077B1', '#BDB8D9', '#9DCEC8', '#EA8575' , '#D12627']

def plot_score_box(files_score, time_indexes):
    # pick the specific time indexes and transform the raw data into dataframe
    score = {}
    for key, val in files_score.items():
        with open(val, 'rb') as file:
            dir = os.path.dirname(val)
            score[key] =  pickle.load(file)['score'][:, time_indexes]
    df = data_to_dict(score, time_indexes)

    # boxplot
    fig, ax = plt.subplots()
    fig.set_size_inches(8,6)
    sns.violinplot(ax=ax, data=df, x="time", y="score", hue="predictor", fill=False, gap=.1)
    set_figure_formats(ax=ax)

    # storage
    fig.savefig(os.path.join(dir, 'score_box.pdf'), bbox_inches='tight')

def plot_score_avg_box(files_score, time_indexes):
    # pick the specific time indexes and transform the raw data into dataframe
    
    score_avg = {}
    for key, val in files_score.items():
        with open(val, 'rb') as file:
            dir = os.path.dirname(val)
            score_avg[key] = cum_average(pickle.load(file)['score'][:, 1:])[:, time_indexes]
    df_avg = data_to_dict(score_avg, time_indexes)

    # boxplot
    fig, ax = plt.subplots()
    fig.set_size_inches(8,6)
    my_pal = {'nominal':'tab:blue', 'drpp': 'tab:orange', 'optimal':'tab:green'}
    sns.boxplot(data=df_avg, x="time", y="score", palette=my_pal, hue="predictor", gap=.1, showmeans=True, meanprops={'markerfacecolor':'white', 'markeredgecolor':'black', 'markersize':'10'})
    set_figure_formats(ax=ax)
    # ax.set_ylim([-10, -2])
    ax.legend(loc='lower right', fontsize=14)
    # ax.set_ylim([-16, -2])
    # ax.legend(loc='lower left', fontsize=14)
    # storage
    fig.savefig(os.path.join(dir, 'score_avg_box.pdf'), bbox_inches='tight')