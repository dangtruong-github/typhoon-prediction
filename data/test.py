import pandas as pd
import xarray as xr

def main():
    path_test = "/N/scratch/tnn3/dataTotal/merra2_preprocessed_rsync/merra2_20221231_18_00.nc"
    ds = xr.open_dataset(path_test)

    longitude_filter = (ds["longitude"] >= 100) & (ds["longitude"] <= 150)
    latitude_filter = (ds["latitude"] >= 0) & (ds["latitude"] <= 30)

    T_VAR = ['RH', 'T', 'T', 'H', 'OMEGA', 'U', 'U', 'V', 'V']
    T_IDX = [10, 4, 16, 16, 16, 8, 22, 8, 22]
    input_arr = []
    for var, idx in zip(T_VAR[:], T_IDX[:]):
        arr = ds.variables[var].data[idx]

        arr = arr[latitude_filter, :][:, longitude_filter]
        
        print(arr.shape)
        input_arr.append(arr)

    print(ds["latitude"])

if __name__ == "__main__":
    main()