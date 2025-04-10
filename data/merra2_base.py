from torch.utils.data import Dataset, WeightedRandomSampler
import pandas as pd
from pathlib import Path
from abc import ABC, abstractmethod
from typing import List
import numpy as np
import os

from data.utils import convert_timestamp_to_filename

def undersample_data(data, label_column="Label", ratio=10):

    data_minority = data[data[label_column] == 1]
    data_majority = data[data[label_column] == 0]
    
    target_majority_count = int(len(data_minority) * ratio)
    data_majority_sampled = data_majority.sample(n=target_majority_count, random_state=42)
    
    undersampled_data = pd.concat([data_minority, data_majority_sampled], axis=0)
    undersampled_data = undersampled_data.sample(frac=1, random_state=42).reset_index(drop=True)
    
    return undersampled_data


class MerraBasePast(Dataset, ABC):
    def __init__(self,
        type_dataset: str, merra_path: Path, agg_step: int = 0,
        agg_alpha: float = 0.85, pos_steps: List[int] | np.array = [0],
        neg_steps: List[int] | np.array = np.arange(20, 41)
    ):        
        if type_dataset not in ["train", "val", "test"]:
            raise ValueError(f"type_dataset={type_dataset} invalid. Try either 'train', 'val', or 'test'")

        self.df = pd.read_csv(merra_path)

        self.agg_step = agg_step
        self.agg_alpha = agg_alpha
        self.weights = np.array([agg_alpha ** i for i in range(agg_step+1)])

        # loc only past domain and positive
        self.df = self.df[self.df['Domain'].isin(["POSITIVE", "Past"])]

        # get timestep and label
        def create_label(step):
            if step in neg_steps:
                return 0
            elif step in pos_steps:
                return 1
            else:
                return -1

        self.df['Label'] = self.df['Step'].apply(create_label)
        self.df = self.df.loc[self.df['Label'].isin([0, 1])]

        # train val test by year
        if type_dataset == "train":
            self.df = self.df.loc[self.df['Year'].isin(np.arange(1999, 2016))]
        elif type_dataset == "train":
            self.df = self.df.loc[self.df['Year'].isin(np.arange(2016, 2019))]
        else:
            self.df = self.df.loc[self.df['Year'].isin(np.arange(2019, 2022))]

        # final check
        self.df = self.df.reset_index(drop=True)

    @abstractmethod
    def load_data_each(self, nc_path):
        pass

    def load_data(self, nc_path, nc_step):
        list_paths = [nc_path]

        res_arr = []

        if self.agg_step > 0:
            search_field = nc_path[:-3].split("_")[-1]
            cond1 = self.df["FileName"].str.contains(search_field)
            cond2 = self.df["Step"].isin(np.arange(
                nc_step+1, min(nc_step+self.agg_step+1, 41)
            ))
            search_data = self.df.loc[cond1 & cond2]

            if search_data.empty:
                return self.load_data_each(nc_path)

            cur_step = nc_step+1
            while len(list_paths) < self.agg_step+1:
                next_path_data = search_data.loc[search_data["Step"] == cur_step]
                cur_step += 1

                if next_path_data.empty:
                    list_paths.append("/N/slate/tnn3/dummy.nc")
                    continue
                
                next_path = next_path_data.reset_index(drop=True).at[0, "Path"]

                list_paths.append(next_path)

        for path_nc in list_paths:
            res_arr.append(self.load_data_each(path_nc))

        res_arr = np.stack(res_arr, axis=0)
        # print(f"res_arr: {res_arr.shape}")

        res = (res_arr * self.weights[:, np.newaxis, np.newaxis, np.newaxis]).sum(axis=0)
        # print(f"res: {res.shape}")

        # assert 0 == 1

        return res

    def __len__(self):
        return len(self.df)
            
    # Load data and agg into a sample
    def __getitem__(self, idx):
        label = self.df.at[idx, "Label"]
        data_path = self.df.at[idx, "Path"]
        cur_step = self.df.at[idx, "Step"]
        
        data = self.load_data(data_path, cur_step)

        data = data.astype('float32')
        
        return data, label


class MerraBaseFullMap(Dataset, ABC):
    def __init__(self,
        type_dataset: str, merra_path: Path, agg_step: int = 0,
        agg_alpha: float = 0.85, pos_steps: List[int] | np.array = [0],
        neg_steps: List[int] | np.array = np.arange(20, 41),
        rus: float = 2
    ):        
        if type_dataset not in ["train", "val", "test"]:
            raise ValueError(f"type_dataset={type_dataset} invalid. Try either 'train', 'val', or 'test'")

        self.df = pd.read_csv(merra_path)

        self.agg_step = agg_step
        self.agg_alpha = agg_alpha
        self.weights = np.array([agg_alpha ** i for i in range(agg_step+1)])
        self.pos_ind = pos_steps[0]
        self.rus = rus

        self.setup_ib(type_dataset)

        # data["Path"] = data["Path"].str.replace("nasa-merra2", "nasa-merra2.old")
        
        self.df.loc[self.df['Label'] != 1, 'Label'] = 0

        # train val test by year
        if type_dataset == "train":
            self.df = self.df.loc[self.df['Year'].isin(np.arange(1980, 2011))]
        elif type_dataset == "val":
            self.df = self.df.loc[self.df['Year'].isin(np.arange(2011, 2017))]
        else:
            self.df = self.df.loc[self.df['Year'].isin(np.arange(2017, 2023))]

        # final check
        self.df = self.df.reset_index(drop=True)

        print(f"self.rus: {self.rus}, {type(self.rus)}")

        if type_dataset != "test":
            self.df = undersample_data(self.df, ratio=self.rus)

        # print(self.df)
        # print(self.pos_ind)

        # os.makedirs("/N/slate/tnn3/TruongChu/merraRun/datasets/csv/test/", exist_ok=True)
        # self.df.to_csv("/N/slate/tnn3/TruongChu/merraRun/datasets/csv/test/pos_{}.csv".format(self.pos_ind))
        # assert 0 == 1

    def setup_ib(self, type_dataset):
        ibtracs_file = '/N/slate/tnn3/TruongChu/merraRun/datasets/csv/FIRST_MERRA2_IBTRACS.csv'
        ibtracs_data = pd.read_csv(ibtracs_file)

        ibtracs_data = ibtracs_data[(ibtracs_data['LAT'] >= 0) &
                        (ibtracs_data['LAT'] <= 30) &
                        (ibtracs_data['LON'] >= 100) &
                        (ibtracs_data['LON'] <= 150)]
        
        self.ibtracs_filenames = set(
        ibtracs_data['ISO_TIME'].apply(lambda x: convert_timestamp_to_filename(
            x, time_steps_back=self.pos_ind
        )))

        if type_dataset == "test":
            self.df['Label'] = 0
            matching_indices = self.df.index[self.df['Filename'].isin(self.ibtracs_filenames)]
            for idx in matching_indices:
                storm_idx = min(idx + self.pos_ind, len(self.df) - 1)
                self.df.loc[idx:storm_idx, 'Label'] = 1
        else:
            self.df['Label'] = np.where(self.df['Filename'].isin(self.ibtracs_filenames), 1, self.df['Label'])
        

    @abstractmethod
    def load_data_each(self, nc_path):
        pass

    def load_data(self, nc_path, nc_step):
        list_paths = [nc_path]

        res_arr = []

        if False:
            search_field = nc_path[:-3].split("_")[-1]
            cond1 = self.df["FileName"].str.contains(search_field)
            cond2 = self.df["Step"].isin(np.arange(
                nc_step+1, min(nc_step+self.agg_step+1, 41)
            ))
            search_data = self.df.loc[cond1 & cond2]

            if search_data.empty:
                return self.load_data_each(nc_path)

            cur_step = nc_step+1
            while len(list_paths) < self.agg_step+1:
                next_path_data = search_data.loc[search_data["Step"] == cur_step]
                cur_step += 1

                if next_path_data.empty:
                    list_paths.append("/N/slate/tnn3/dummy.nc")
                    continue
                
                next_path = next_path_data.reset_index(drop=True).at[0, "Path"]

                list_paths.append(next_path)

        for path_nc in list_paths:
            res_arr.append(self.load_data_each(path_nc))

        res_arr = np.stack(res_arr, axis=0)
        # print(f"res_arr: {res_arr.shape}")

        res = (res_arr * self.weights[:, np.newaxis, np.newaxis, np.newaxis]).sum(axis=0)
        # print(f"res: {res.shape}")

        # assert 0 == 1

        return res

    def __len__(self):
        return len(self.df)
            
    # Load data and agg into a sample
    def __getitem__(self, idx):
        label = self.df.at[idx, "Label"]
        data_path = self.df.at[idx, "Path"]
        # cur_step = self.df.at[idx, "Step"]
        
        data = self.load_data(data_path, 0)

        data = data.astype('float32')
        
        return data, label
