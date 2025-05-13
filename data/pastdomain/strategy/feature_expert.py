import pandas as pd
import numpy as np
import xarray as xr

from data.utils import (
    meshgrid,
    vorticity, divergence,
)

T_VAR = ['RH', 'T', 'T', 'H', 'OMEGA', 'U', 'U', 'V', 'V', 'VOR', 'VOR', 'VOR', 'DIV']
T_PRS = [750, 900, 500, 500, 500, 800, 200, 800, 200, 900, 700, 200, 200]
T_CAT = ['RH_750', 'T_900', 'T_500', 'H_500', 'OMEGA_500', 'U_800', 'U_200', 'V_800', 'V_200', 'VOR_900', 'VOR_700', 'VOR_200', 'DIV_200']
T_IDX = [10, 4, 16, 16, 16, 8, 22, 8, 22, 4, 12, 22, 22]

def get_mean_std(stat_path):
    if stat_path is None:
        stat_path = '/N/slate/tnn3/TruongChu/merraRun/datasets/csv/data_statistics_fexpert.xlsx'

    stat = pd.read_excel(stat_path)
    stat.set_index('Variable', inplace=True)
    mean = stat['Mean'].loc[T_CAT].to_numpy()[:, np.newaxis, np.newaxis]
    std = stat['Std'].loc[T_CAT].to_numpy()[:, np.newaxis, np.newaxis]

    return mean, std

def get_feature_expert(load_path, stat_path, type_data):
    mean, std = get_mean_std(stat_path)
    try:
        ds = xr.open_dataset(load_path).sel(latitude=slice(0, 30), longitude=slice(100, 150))
    except:
        if type_data == "past":
            return (np.zeros((13, 33, 33)) - mean) / std
        else:
            return (np.zeros((13, 61, 81)) - mean) / std
        
    input_arr = []
    for var, idx in zip(T_VAR[: - 4], T_IDX[: - 4]):
        arr = ds.variables[var].data[idx]
        arr = np.expand_dims(arr, axis=0)
        input_arr.append(arr)
        # print(f"arr.shape: {arr.shape}")

    for idx in T_IDX[- 4: - 1]:
        U = ds.variables['U'].data[idx: idx + 1]
        V = ds.variables['V'].data[idx: idx + 1]
        lon = ds.coords['longitude'].data
        lat = ds.coords['latitude'].data[:: -1]
        
        lat_grid, lon_grid = meshgrid(lat, lon, 1)
        VOR = vorticity(U, V, lat_grid, lon_grid)
        input_arr.append(VOR)
        # print(f"VOR.shape: {VOR.shape}")
        
    for idx in T_IDX[- 1:]:
        U = ds.variables['U'].data[idx: idx + 1]
        V = ds.variables['V'].data[idx: idx + 1]
        lon = ds.coords['longitude'].data
        lat = ds.coords['latitude'].data[:: -1]
        
        lat_grid, lon_grid = meshgrid(lat, lon, 1)
        DIV = divergence(U, V, lat_grid, lon_grid)
        input_arr.append(DIV)
        # print(f"DIV.shape: {DIV.shape}")

    # print(np.array(input_arr).shape)
    # print(mean.shape)
    # print(std.shape)
    concat_data = np.concatenate(input_arr, axis=0)
    
    concat_data[np.isnan(concat_data)] = 0
    res = (concat_data - mean) / std # norm

    return res
