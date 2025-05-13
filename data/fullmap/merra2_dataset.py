import sys
import numpy as np
import pandas as pd
from pathlib import Path
import os

import torch
from torch.utils.data import Dataset

if __name__ == "__main__":
    sys.path.insert(0, '/N/slate/tnn3/TruongChu/merraRun/')

from data.utils import get_save_path
from data.fullmap.strategy.feature_expert import get_feature_expert
from data.fullmap.strategy.full import get_full


device = "cuda" if torch.cuda.is_available() else "cpu"

class Merra2_Dataset(Dataset):
    def __init__(self, 
        data_df: pd.DataFrame, agg_step: int = 0,
        agg_alpha: float = 0.85, stat_path: Path = None,
        type_data: str = "past", type_retrieve: str = "expert"
    ):
        if type_data not in ["past", "fullmap"]:
            raise ValueError(f"Type data {type_data} not allowed. Try either 'past' or 'fullmap'.")
        if type_retrieve not in ["expert", "full"]:
            raise ValueError(f"Retrive data strategy {type_retrieve} not allowed. Try either 'expert' or 'full'.")
        
        # print("HERE")
        # assert 0 == 1
        self.df = data_df
        self.agg_step = agg_step
        self.agg_alpha = agg_alpha
        self.stat_path = stat_path
        self.type_data = type_data
        self.type_retrieve = type_retrieve
        self.weights = np.array([agg_alpha ** i for i in range(agg_step+1)])
        
        if type_retrieve == "expert":
            # print("HERE EXPERT")
            self.func_retrieve = get_feature_expert
        elif type_retrieve == 'full':
            # print("HERE FULL")
            self.func_retrieve = get_full

    def load_data_each(self, nc_path):
        # print("ALABAMA")
        load_path = get_save_path(nc_path)
        if os.path.isfile(load_path) is False:
            print("WE ARE HERE FILE AINT EXIST")
            # print(f"type_retrieve: {self.type_retrieve}")
            return_data = self.func_retrieve("dummy.pt", self.stat_path)

            return return_data
        
        """
        load_path, save_path = get_save_path(nc_path, type_save=self.type_data)

        if os.path.isfile(save_path):
            try:
                res = np.load(save_path)
                # return res
                return res
            except:
                os.remove(save_path)

        print(load_path)
        """

        res = self.func_retrieve(load_path, self.stat_path)

        # print(res.shape)
        
        # print(save_path)

        # assert 1 == 0
        # np.save(save_path, res)

        return res

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
        file_name = self.df.at[idx, "Filename"]
        data_path = self.df.at[idx, "Path"]
        # cur_step = self.df.at[idx, "Step"]
        
        data = self.load_data(data_path, 0)

        data = data.astype('float32')
        
        return {
            "data": data,
            "label": label,
            "file_name": file_name
        }

if __name__ == "__main__":
    path_test = "/N/scratch/tnn3/dataTotal/merra2_preprocessed_rsync/merra2_20221231_18_00.nc"
    dataset_test = Merra2_Dataset()

    print(len(dataset_test.df))
    
    res, _ = dataset_test.__getitem__(0)

    print(res.shape)
