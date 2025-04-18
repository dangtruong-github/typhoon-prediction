import xarray as xr
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List
import lightning as L
import os
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from data.utils import (
    convert_timestamp_to_filename,
    undersample_data
)

if __name__ == "__main__":
    sys.path.insert(0, '/N/slate/tnn3/TruongChu/merraRun/')

from data.utils import (
    meshgrid,
    vorticity, divergence,
    get_save_path
)
from data.merra2_base import MerraBaseFullMap

device = "cuda" if torch.cuda.is_available() else "cpu"

class MerraFeatureExpertFullMap(MerraBaseFullMap):
    def __init__(self, 
        data_df: pd.DataFrame, agg_step: int = 0,
        agg_alpha: float = 0.85, stat_path: Path = None
    ):
        # print("HERE")
        # assert 0 == 1

        super().__init__(data_df, agg_step, agg_alpha)

        if stat_path is None:
            stat_path = '/N/slate/tnn3/TruongChu/merraRun/datasets/csv/data_statistics_fexpert.xlsx'

        self.T_VAR = ['RH', 'T', 'T', 'H', 'OMEGA', 'U', 'U', 'V', 'V', 'VOR', 'VOR', 'VOR', 'DIV']
        self.T_PRS = [750, 900, 500, 500, 500, 800, 200, 800, 200, 900, 700, 200, 200]
        self.T_CAT = ['RH_750', 'T_900', 'T_500', 'H_500', 'OMEGA_500', 'U_800', 'U_200', 'V_800', 'V_200', 'VOR_900', 'VOR_700', 'VOR_200', 'DIV_200']
        self.T_IDX = [10, 4, 16, 16, 16, 8, 22, 8, 22, 4, 12, 22, 22]

        self.stat = pd.read_excel(stat_path)
        self.stat.set_index('Variable', inplace=True)
        self.mean = self.stat['Mean'].loc[self.T_CAT].to_numpy()[:, np.newaxis, np.newaxis]
        self.std = self.stat['Std'].loc[self.T_CAT].to_numpy()[:, np.newaxis, np.newaxis]

    def load_data_each(self, nc_path):
        if os.path.isfile(nc_path) is False:
            return_data = (np.zeros((13, 61, 81)) - self.mean) / self.std

            return return_data
        
        load_path, save_path = get_save_path(nc_path, type_save="full_map")

        if os.path.isfile(save_path):
            try:
                res = np.load(save_path)
                # return res
                return res
            except:
                os.remove(save_path)

        print(load_path)
        ds = xr.open_dataset(load_path).sel(latitude=slice(0, 30), longitude=slice(100, 150))
        
        input_arr = []
        for var, idx in zip(self.T_VAR[: - 4], self.T_IDX[: - 4]):
            arr = ds.variables[var].data[idx]
            arr = np.expand_dims(arr, axis=0)
            input_arr.append(arr)
            # print(f"arr.shape: {arr.shape}")

        for idx in self.T_IDX[- 4: - 1]:
            U = ds.variables['U'].data[idx: idx + 1]
            V = ds.variables['V'].data[idx: idx + 1]
            lon = ds.coords['longitude'].data
            lat = ds.coords['latitude'].data[:: -1]
            
            lat_grid, lon_grid = meshgrid(lat, lon, 1)
            VOR = vorticity(U, V, lat_grid, lon_grid)
            input_arr.append(VOR)
            # print(f"VOR.shape: {VOR.shape}")
            
        for idx in self.T_IDX[- 1:]:
            U = ds.variables['U'].data[idx: idx + 1]
            V = ds.variables['V'].data[idx: idx + 1]
            lon = ds.coords['longitude'].data
            lat = ds.coords['latitude'].data[:: -1]
            
            lat_grid, lon_grid = meshgrid(lat, lon, 1)
            DIV = divergence(U, V, lat_grid, lon_grid)
            input_arr.append(DIV)
            # print(f"DIV.shape: {DIV.shape}")
    
        # print(np.array(input_arr).shape)
        # print(self.mean.shape)
        # print(self.std.shape)
        concat_data = np.concatenate(input_arr, axis=0)
        concat_data[np.isnan(concat_data)] = 0
        res = (concat_data - self.mean) / self.std # norm
        # res[np.isnan(res)] = 0

        # print(res.shape)
        print(save_path)

        # assert 1 == 0
        # np.save(save_path, res)

        return res

class MerraFeatureExpertFullMapModule(L.LightningDataModule):
    def __init__(self,
        folder_save: str, batch_size: int=64,
        rate_under_sampling: float=1, agg_step: int=0,
        agg_alpha: float=0.85, pos_step: int=0, merra_path: str=None
    ):
        super().__init__()
        if merra_path is None:
            merra_path = "/N/slate/tnn3/TruongChu/merraRun/datasets/csv/merra_full_new_2.csv"

        self.merra_path = merra_path

        self.batch_size = batch_size
        self.rus = rate_under_sampling
        self.agg_step = agg_step
        self.agg_alpha = agg_alpha
        self.pos_step = pos_step

        print(f"pos_step: {pos_step}")
        print(f"rate_under_sampling: {rate_under_sampling}")
        
        self.num_workers = 32 if device == "cuda" else 1

        self.preset_ib()
        self.prepare_df()
        self.setup()

        self.df_train.to_csv(os.path.join(folder_save, "train_set.csv"), index=False)
        self.df_val.to_csv(os.path.join(folder_save, "val_set.csv"), index=False)
        self.df_test.to_csv(os.path.join(folder_save, "test_set.csv"), index=False)

    def prepare_df(self):
        df_train_val = self.split_set("train_val")
        self.df_train, self.df_val = train_test_split(df_train_val, test_size=0.1, random_state=42)
        self.df_test = self.split_set("test")

        self.df_train = self.df_train.sort_values(by="Filename").reset_index(drop=True)
        self.df_val = self.df_val.sort_values(by="Filename").reset_index(drop=True)
        self.df_test = self.df_test.sort_values(by="Filename").reset_index(drop=True)

    def split_set(self, type_dataset):
        df = self.setup_ib(type_dataset)

        # data["Path"] = data["Path"].str.replace("nasa-merra2", "nasa-merra2.old")
        
        df.loc[df['Label'] != 1, 'Label'] = 0

        # train val test by year
        if type_dataset == "train_val":
            df = df.loc[df['Year'].isin(np.arange(1980, 2017))]
        else:
            df = df.loc[df['Year'].isin(np.arange(2017, 2023))]

        # final check
        df = df.reset_index(drop=True)

        print(f"self.rus: {self.rus}, {type(self.rus)}")

        if type_dataset != "test":
            df = undersample_data(df, ratio=self.rus)

        return df

    def preset_ib(self):
        ibtracs_file = '/N/slate/tnn3/TruongChu/merraRun/datasets/csv/FIRST_MERRA2_IBTRACS.csv'
        ibtracs_data = pd.read_csv(ibtracs_file)

        ibtracs_data = ibtracs_data[(ibtracs_data['LAT'] >= 0) &
                        (ibtracs_data['LAT'] <= 30) &
                        (ibtracs_data['LON'] >= 100) &
                        (ibtracs_data['LON'] <= 150)]
        
        self.ibtracs_filenames = set(
        ibtracs_data['ISO_TIME'].apply(lambda x: convert_timestamp_to_filename(
            x, time_steps_back=self.pos_step
        )))

    def setup_ib(self, type_dataset):
        df = pd.read_csv(self.merra_path)

        if type_dataset == "test":
            df['Label'] = 0
            matching_indices = df.index[df['Filename'].isin(self.ibtracs_filenames)]
            for idx in matching_indices:
                storm_idx = min(idx + self.pos_step, len(df) - 1)
                df.loc[idx:storm_idx, 'Label'] = 1
        else:
            df['Label'] = np.where(df['Filename'].isin(self.ibtracs_filenames), 1, df['Label'])

        return df

    def setup(self, stage=None):
        self.train_set = MerraFeatureExpertFullMap(
            data_df=self.df_train, agg_step=self.agg_step,
            agg_alpha=self.agg_alpha
        )
        self.val_set = MerraFeatureExpertFullMap(
            data_df=self.df_val, agg_step=self.agg_step,
            agg_alpha=self.agg_alpha
        )
        self.test_set = MerraFeatureExpertFullMap(
            data_df=self.df_test, agg_step=self.agg_step,
            agg_alpha=self.agg_alpha
        )
        
        rus_weights = [self.rus if label == 1 else 1 for label in self.train_set.df["Label"]]

        self.rus_sampler = WeightedRandomSampler(rus_weights, num_samples=len(rus_weights))

    def train_dataloader(self):
        return DataLoader(
            self.train_set, batch_size=self.batch_size,
            # sampler=self.rus_sampler,
            num_workers=self.num_workers
        )

    def train_dataloader_random(self):
        return DataLoader(
            self.train_set, batch_size=self.batch_size,
            num_workers=self.num_workers
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
    dataset_test = MerraFeatureExpertFullMap()

    print(len(dataset_test.df))
    
    res, _ = dataset_test.__getitem__(0)

    print(res.shape)
