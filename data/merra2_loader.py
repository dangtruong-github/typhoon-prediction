import sys
import numpy as np
import pandas as pd
import lightning as L
import os
from sklearn.model_selection import train_test_split
import multiprocessing

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

if __name__ == "__main__":
    sys.path.insert(0, '/N/slate/tnn3/TruongChu/merraRun/')

from data.utils import (
    convert_timestamp_to_filename,
    undersample_data
)
from data.merra2_dataset import Merra2_Dataset

device = "cuda" if torch.cuda.is_available() else "cpu"
num_workers = min(32, multiprocessing.cpu_count())

class Merra2_fullmap_Loader(L.LightningDataModule):
    def __init__(self,
        folder_save: str, batch_size: int=64,
        rate_under_sampling: float=1, agg_step: int=0,
        agg_alpha: float=0.85, pos_step: int=0, merra_path: str=None,
        type_retrieve: str = "expert"
    ):
        if type_retrieve not in ["expert", "full"]:
            raise ValueError(f"Retrive data strategy {type_retrieve} not allowed. Try either 'expert' or 'full'.")

        super().__init__()
        if merra_path is None:
            merra_path = "/N/slate/tnn3/TruongChu/merraRun/datasets/csv/merra_full_new_2.csv"

        self.merra_path = merra_path

        self.batch_size = batch_size
        self.rus = rate_under_sampling
        self.agg_step = agg_step
        self.agg_alpha = agg_alpha
        self.pos_step = pos_step
        self.type_data = "fullmap"
        self.type_retrieve = type_retrieve

        print(f"pos_step: {pos_step}")
        print(f"rate_under_sampling: {rate_under_sampling}")

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
        self.train_set = Merra2_Dataset(
            data_df=self.df_train, agg_step=self.agg_step,
            agg_alpha=self.agg_alpha, type_data=self.type_data,
            type_retrieve=self.type_retrieve
        )
        self.val_set = Merra2_Dataset(
            data_df=self.df_val, agg_step=self.agg_step,
            agg_alpha=self.agg_alpha, type_data=self.type_data,
            type_retrieve=self.type_retrieve
        )
        self.test_set = Merra2_Dataset(
            data_df=self.df_test, agg_step=self.agg_step,
            agg_alpha=self.agg_alpha, type_data=self.type_data,
            type_retrieve=self.type_retrieve
        )
        
        rus_weights = [self.rus if label == 1 else 1 for label in self.train_set.df["Label"]]

        self.rus_sampler = WeightedRandomSampler(rus_weights, num_samples=len(rus_weights))

    def train_dataloader(self):
        return DataLoader(
            self.train_set, batch_size=self.batch_size,
            num_workers=num_workers
        )

    def train_dataloader_random(self):
        return DataLoader(
            self.train_set, batch_size=self.batch_size,
            sampler=self.rus_sampler,
            num_workers=num_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set, batch_size=self.batch_size,
            num_workers=num_workers
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_set, batch_size=self.batch_size,
            num_workers=num_workers
        )

class Merra2_past_Loader(L.LightningDataModule):
    def __init__(self,
        folder_save: str, batch_size: int=64,
        rate_under_sampling: float=1, agg_step: int=0,
        agg_alpha: float=0.85, pos_step: int=0, merra_path: str=None,
        type_retrieve: str = "expert"
    ):
        if type_retrieve not in ["expert", "full"]:
            raise ValueError(f"Retrive data strategy {type_retrieve} not allowed. Try either 'expert' or 'full'.")

        super().__init__()
        if merra_path is None:
            merra_path = "/N/slate/tnn3/TruongChu/merraRun/datasets/data_path.csv"

        self.merra_path = merra_path

        self.batch_size = batch_size
        self.rus = rate_under_sampling
        self.agg_step = agg_step
        self.agg_alpha = agg_alpha
        self.pos_step = pos_step
        self.type_data = "past"
        self.type_retrieve = type_retrieve

        print(f"pos_step: {pos_step}")
        print(f"rate_under_sampling: {rate_under_sampling}")

        self.prepare_df()
        self.setup()

        self.df_train.to_csv(os.path.join(folder_save, "train_set.csv"), index=False)
        self.df_val.to_csv(os.path.join(folder_save, "val_set.csv"), index=False)
        self.df_test.to_csv(os.path.join(folder_save, "test_set.csv"), index=False)

    def prepare_df(self):
        # Step 1: Filter Domain and Noise
        df = pd.read_csv(self.merra_path)
        df_filtered = df[
            (df['Domain'].isin(['Past', 'POSITIVE'])) &
            (df['Noise'].isnull() | (df['Noise'] == 0))
        ]

        # Step 2: Label based on Step
        df_filtered = df_filtered.copy()  # avoid SettingWithCopyWarning
        df_filtered['Label'] = -1  # default

        df_filtered.loc[df_filtered['Step'] == self.pos_step, 'Label'] = 1
        df_filtered.loc[df_filtered['Step'].between(20, 40), 'Label'] = 0

        # Remove rows that do not belong to either class
        df_filtered = df_filtered[df_filtered['Label'] != -1]

        # Step 3: Train/Val and Test Split by Year
        train_val_df = df_filtered[df_filtered['Year'].between(1980, 2016)]
        self.df_test = df_filtered[df_filtered['Year'].between(2017, 2022)]

        # Step 4: Train/Val split (90% train, 10% val)
        self.df_train, self.df_val = train_test_split(train_val_df, test_size=0.1, stratify=train_val_df['Label'], random_state=42)

        rename_map = {'FileName': 'Filename'}

        self.df_train = self.df_train.rename(columns=rename_map)
        self.df_val = self.df_val.rename(columns=rename_map)
        self.df_test = self.df_test.rename(columns=rename_map)

        self.df_train = self.df_train.reset_index(drop=True)
        self.df_val = self.df_val.reset_index(drop=True)
        self.df_test = self.df_test.reset_index(drop=True)

    def setup(self, stage=None):
        self.train_set = Merra2_Dataset(
            data_df=self.df_train, agg_step=self.agg_step,
            agg_alpha=self.agg_alpha, type_data=self.type_data,
            type_retrieve=self.type_retrieve
        )
        self.val_set = Merra2_Dataset(
            data_df=self.df_val, agg_step=self.agg_step,
            agg_alpha=self.agg_alpha, type_data=self.type_data,
            type_retrieve=self.type_retrieve
        )
        self.test_set = Merra2_Dataset(
            data_df=self.df_test, agg_step=self.agg_step,
            agg_alpha=self.agg_alpha, type_data=self.type_data,
            type_retrieve=self.type_retrieve
        )
        
        rus_weights = [self.rus if label == 1 else 1 for label in self.train_set.df["Label"]]

        self.rus_sampler = WeightedRandomSampler(rus_weights, num_samples=len(rus_weights))

    def train_dataloader(self):
        return DataLoader(
            self.train_set, batch_size=self.batch_size,
            num_workers=num_workers
        )

    def train_dataloader_random(self):
        return DataLoader(
            self.train_set, batch_size=self.batch_size,
            sampler=self.rus_sampler,
            num_workers=num_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set, batch_size=self.batch_size,
            num_workers=num_workers
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_set, batch_size=self.batch_size,
            num_workers=num_workers
        )
