import torch
import os
import numpy as np
import pandas as pd

SINGLE_VAR = ['PHIS', 'PS', 'SLP',]
PRESS_VAR = ['H', 'OMEGA', 'QI', 'QL', 'QV', 'RH', 'T', 'U', 'V']

# SINGLE_VAR = ['PHIS', 'PS',]
# PRESS_VAR = ['QI', 'QL', 'QV', 'U', 'V']

LEVEL = 25

PRESS_LEVEL = [1000, 975, 950, 925, 900, 875, 850, 825, 
               800, 775, 750, 725, 700, 650, 600, 550, 
               500, 450, 400, 350, 300, 250, 200, 150, 100]

# number of channels after concat
INP_CHANNELS = len(SINGLE_VAR) + LEVEL * len(PRESS_VAR)

LIST_VAR = [var + '0' for var in SINGLE_VAR]
LIST_VAR.extend([var + str(level) for var in PRESS_VAR for level in PRESS_LEVEL])

def get_mean_std(stat_path):
    if stat_path is None:
        stat_path = '/N/slate/tnn3/TruongChu/merraRun/datasets/data_train_stats.xlsx'

    stat = pd.read_excel(stat_path)
    stat['variable_name'] = stat['variable'] + stat['level'].astype(str)
    stat.set_index('variable_name', inplace=True)
    mean = stat['mean'].loc[LIST_VAR].to_numpy()[:, np.newaxis, np.newaxis]
    std = stat['std'].loc[LIST_VAR].to_numpy()[:, np.newaxis, np.newaxis]

    return mean, std

def get_full(load_path, stat_path):
    mean, std = get_mean_std(stat_path)

    if os.path.isfile(load_path) is False:
        return (np.zeros((228, 61, 81)) - mean) / std
    
    data = torch.load(load_path)  # Load the tensor
    numpy_array = data.numpy()  # Convert to NumPy array
    return (numpy_array - mean) / std
