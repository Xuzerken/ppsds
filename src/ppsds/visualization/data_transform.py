"""Small reshaping helpers shared by the plotting modules."""

import numpy as np
import pandas as pd

def cum_average(data):
    """Cumulative average along axis 1 (step axis)."""
    data = np.cumsum(data, axis=1)
    for i in range(data.shape[1]):
        data[:, i] /= i+1
    return data

def data_to_dict(score, time_indexes):
    '''
    INPUTS:
    - score is a dictionary with each key being the predictor name, and the according value is its score matrix (traj_num * time_indexes)
    - time_indexes is a list of the choosen time indexes to be plotted

    OUTPUTS:
    - each row is the score of certain predictor at certain time indexes on certain trajectory
    - the columns include predictor, score and time
    '''
    dfs = []
    for key, val in score.items():
        dic = {
            'score': val.flatten(),
            'time': time_indexes * val.shape[0],
            'predictor': key
        }
        dfs.append(pd.DataFrame(dic))
    return pd.concat(dfs, ignore_index=True, axis=0)
