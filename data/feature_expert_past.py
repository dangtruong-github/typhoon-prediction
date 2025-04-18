import xarray as xr
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List
import lightning as L
import os

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

if __name__ == "__main__":
    sys.path.insert(0, '/N/slate/tnn3/TruongChu/merraRun/')

from data.utils import (
    meshgrid, vorticity, divergence,
    get_save_path
)
from data.merra2_base import MerraBasePast

device = "cuda" if torch.cuda.is_available() else "cpu"

class MerraFeatureExpertPast(MerraBasePast):
    def __init__(self, 
        merra_path: Path = None, agg_step: int = 0,
        agg_alpha: float = 0.85, pos_steps: List[int] | np.array = [0],
        neg_steps: List[int] | np.array = np.arange(20, 41),
        stat_path: Path = None, type_dataset: str = "train",
        pos_step: int = 0
    ):
        if merra_path is None:
            merra_path = "/N/slate/tnn3/TruongChu/merraRun/datasets/merra_path.csv"
        pos_steps = [pos_step]

        super().__init__(type_dataset, merra_path, agg_step, agg_alpha, pos_steps, neg_steps)
        self.T_VAR = ['RH', 'T', 'T', 'H', 'OMEGA', 'U', 'U', 'V', 'V', 'VOR', 'VOR', 'VOR', 'DIV']
        self.T_PRS = [750, 900, 500, 500, 500, 800, 200, 800, 200, 900, 700, 200, 200]
        self.T_CAT = ['RH750', 'T900', 'T500', 'H500', 'OMEGA500', 'U800', 'U200', 'V800', 'V200', 'VOR900', 'VOR700', 'VOR200', 'DIV200']
        self.T_IDX = [10, 4, 16, 16, 16, 8, 22, 8, 22, 4, 12, 22, 22]

        if stat_path is None:
            stat_path = '/N/slate/tnn3/TruongChu/merraRun/datasets/data_train_statistics.xlsx'

        self.stat = pd.read_excel(stat_path)
        self.stat['variable_name'] = self.stat['variable'] + self.stat['level'].astype(str)
        self.stat.set_index('variable_name', inplace=True)
        self.mean = self.stat['mean'].loc[self.T_CAT].to_numpy()[:, np.newaxis, np.newaxis]
        self.std = self.stat['std'].loc[self.T_CAT].to_numpy()[:, np.newaxis, np.newaxis]

    def load_data_each(self, nc_path):
        if os.path.isfile(nc_path) is False:
            return np.zeros((13, 33, 33))
        
        save_path = get_save_path(nc_path)

        if os.path.isfile(save_path):
            res = np.load(save_path)
            return res

        ds = xr.open_dataset(nc_path)
        input_arr = []
        for var, idx in zip(self.T_VAR[: - 4], self.T_IDX[: - 4]):
            arr = ds.variables[var].data[idx]
            input_arr.append(arr)
            
        for idx in self.T_IDX[- 4: - 1]:
            U = ds.variables['U'].data[idx: idx + 1]
            V = ds.variables['V'].data[idx: idx + 1]
            lon = ds.coords['longitude'].data
            lat = ds.coords['latitude'].data[:: -1]
            
            lat_grid, lon_grid = meshgrid(lat, lon, 1)
            VOR = vorticity(U, V, lat_grid, lon_grid)
            input_arr.extend(VOR)
            
        for idx in self.T_IDX[- 1:]:
            U = ds.variables['U'].data[idx: idx + 1]
            V = ds.variables['V'].data[idx: idx + 1]
            lon = ds.coords['longitude'].data
            lat = ds.coords['latitude'].data[:: -1]
            
            lat_grid, lon_grid = meshgrid(lat, lon, 1)
            DIV = divergence(U, V, lat_grid, lon_grid)
            input_arr.extend(DIV)

        res = (np.array(input_arr) - self.mean) / self.std # norm
        res[np.isnan(res)] = 0

        # print(res.shape)
        np.save(save_path, res)

        return res

class MerraFeatureExpertPastModule(L.LightningDataModule):
    def __init__(self,
        batch_size=64, rate_under_sampling: float = 1,
        agg_step: int=0, pos_step: int=0
    ):
        super().__init__()
        self.batch_size = batch_size
        self.rate_under_sampling = rate_under_sampling
        self.agg_step = agg_step
        self.pos_step = pos_step

        self.setup()

    def setup(self, stage=None):
        self.train_set = MerraFeatureExpertPast(
            type_dataset="train", agg_step=self.agg_step, pos_step=self.pos_step
        )
        self.val_set = MerraFeatureExpertPast(
            type_dataset="val", agg_step=self.agg_step, pos_step=self.pos_step
        )
        self.test_set = MerraFeatureExpertPast(
            type_dataset="test", agg_step=self.agg_step, pos_step=self.pos_step
        )

        rus_weights = [self.rate_under_sampling if label == 1 else 1 for label in self.train_set.df["Label"]]

        self.rus_sampler = WeightedRandomSampler(rus_weights, num_samples=len(rus_weights))
        
        self.num_workers = 32 if device == "cuda" else 1

    def train_dataloader(self):
        return DataLoader(
            self.train_set, batch_size=self.batch_size,
            sampler=self.rus_sampler, num_workers=self.num_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set, batch_size=self.batch_size,
            num_workers=self.num_workers
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_set, batch_size=self.batch_size,
            num_workers=self.num_workers
        )

if __name__ == "__main__":
    path_test = "/N/scratch/tnn3/dataTotal/merra2_preprocessed_rsync/merra2_20221231_18_00.nc"
    dataset_test = MerraFeatureExpertPast()

    print(len(dataset_test.df))
    
    res, _ = dataset_test.__getitem__(0)

    print(res.shape)
