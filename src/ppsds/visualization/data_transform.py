"""Small reshaping helpers shared by the plotting modules."""

import pandas as pd

from ..predictability.trajectory import cumulative_average

def cum_average(data):
    """Cumulative average along axis 1 (step axis).

    Thin alias for the core implementation, which the predictability figures use too -- one
    definition, so the core and the plotting periphery cannot drift apart.
    """
    return cumulative_average(data)

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
