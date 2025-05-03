import numpy as np
import os
from datetime import datetime, timedelta
import pandas as pd

omega = 7.29 * 1e-5

def vorticity(u, v, lat, lon):
    x = lon# * 111 * 1000
    y = lat# * 111 * 1000
    lat = np.deg2rad(lat)
    
    vx = np.divide(v, x, out=np.full_like(v, np.nan, dtype=np.float64), where=(x != 0))
    uy = np.divide(u, y, out=np.full_like(u, np.nan, dtype=np.float64), where=(y != 0))
    
    return vx - uy + 2 * omega * np.sin(lat) * 111 * 1000

def divergence(u, v, lat, lon):
    x = lon# * 111 * 1000
    y = lat# * 111 * 1000
    
    ux = np.divide(u, x, out=np.full_like(u, np.nan, dtype=np.float64), where=(x != 0))
    vy = np.divide(v, y, out=np.full_like(v, np.nan, dtype=np.float64), where=(y != 0))
    
    return ux + vy

def meshgrid(lat, lon, lvl = 1):
    
    lon_grid, lat_grid = np.meshgrid(lon, lat)
    
    lat_grid = np.repeat(lat_grid[np.newaxis, :, :], lvl, axis=0)
    lon_grid = np.repeat(lon_grid[np.newaxis, :, :], lvl, axis=0)
    
    return lat_grid, lon_grid

def lat_lon_grid_fullmap(lat, lon):
    lat_grid, lon_grid = meshgrid(lat, lon, 1)

    lat_filter = (lat_grid[0, :, 0] >= 0) & (lat_grid[0, :, 0] <= 30)
    lon_filter = (lon_grid[0, 0, :] >= 100) & (lon_grid[0, 0, :] <= 150)

    lat_grid = lat_grid[:, lat_filter, :]
    lon_grid = lon_grid[:, :, lon_filter]

    lat_grid = lat_grid[:, :, :81]
    lon_grid = lon_grid[:, :61, :]

    return lat_grid, lon_grid

def get_save_path(nc_path, type_save="past"):
        base_folder = "/N/scratch/tnn3/dataTotal/merra2_rsync/"
        if type_save == "fullmap":
            base_folder = "/N/scratch/tnn3/dataTotal/merra2_preprocessed_rsync/"
        new_folder = "truong/feature_expert"
        extracted_path = "/".join(nc_path.split("/")[-2:]).replace(".nc", ".npy")
        extracted_load_path = nc_path.split("/")[-1]

        save_path = os.path.join(base_folder, new_folder, extracted_path)
        load_path = os.path.join(base_folder, extracted_load_path)

        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        return load_path, save_path

def convert_timestamp_to_filename(timestamp, time_steps_back=0):
    try:
        # Parse the timestamp
        dt = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
        
        # Calculate the new timestamp by going back the specified number of time steps
        # Each time step is 3 hours
        dt -= timedelta(hours=time_steps_back * 3)
        
        # Format into the desired filename
        filename = f"merra2_{dt.strftime('%Y%m%d_%H_00')}.nc"
        return filename
    except ValueError as e:
        print(f"Error parsing timestamp: {timestamp}. Ensure it is in 'YYYY-MM-DD HH:MM:SS' format.")
        return None

def undersample_data(data, label_column="Label", ratio=10):

    data_minority = data[data[label_column] == 1]
    data_majority = data[data[label_column] == 0]
    
    target_majority_count = int(len(data_minority) * ratio)
    data_majority_sampled = data_majority.sample(n=target_majority_count, random_state=42)
    
    undersampled_data = pd.concat([data_minority, data_majority_sampled], axis=0)
    undersampled_data = undersampled_data.sample(frac=1, random_state=42).reset_index(drop=True)
    
    return undersampled_data