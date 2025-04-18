import os
import numpy as np
import argparse
from datetime import datetime
import yaml

from data.total import get_data_module

def check_dirties(data_test, data_compare):
    diff = data_test - data_compare

    dirties = 0

    for var_idx in range(13):
        # print(f"var_idx: {var_idx}")
        for i in range(61):
            for j in range(81):
                if diff[var_idx, i, j] < 1e-6:
                    continue
                    print("-", end="")
                else:
                    # print("X", end="")
                    dirties += 1
            # print()

    """
    for var_idx in range(13):
        print(f"var_idx: {var_idx}")
        for i in range(61):
            for j in range(81):
                if diff[var_idx, i, j] >= 1e-6:
                    print(var_idx, i, j)
                    print(data_test[var_idx, i, j])
                    print(data_compare[var_idx, i, j])
    """

    dirties_pct = dirties / (13*61*81) * 100
    print(f"Percentage of dirties: {dirties_pct}%")

    try:
        assert dirties_pct == 0.0
    except:
        return 0
        for var_idx in range(13):
            print(f"var_idx: {var_idx}")
            for i in range(61):
                for j in range(81):
                    if diff[var_idx, i, j] >= 1e-6:
                        print(var_idx, i, j)
                        print(data_test[var_idx, i, j])
                        print(data_compare[var_idx, i, j])

        raise AssertionError("Dirties!")

def unit_testing_module(data_module):
    base_path = "/N/u/tqluu/BigRed200/@PUBLIC/data_preprocessed/nasa-merra2/"
    unit_test_folder = "/N/slate/tnn3/TruongChu/merraRun/datasets/unit_test"
    for idx, filename in enumerate(os.listdir(unit_test_folder)):
        load_path = os.path.join(base_path, filename.replace(".npy", ".nc"))
        data_test = data_module.train_set.load_data_each(load_path)
        data_test = np.array(data_test, dtype=np.float32)
        data_compare = np.load(os.path.join(unit_test_folder, filename))

        # print(data_test.shape)
        # print(data_compare.shape)
        # print(data_test.dtype)
        # print(data_compare.dtype)

        # print(data_test[:9])
        # print("-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x")
        # print(data_compare[:9])
        # print("-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x")

        # Use np.array_equal for exact match, or np.allclose for approximate comparison
        # assert np.allclose(data_test[:9], data_compare[:9], atol=1e-6), f"Arrays are not approximately equal for {filename} of index {idx} for first 9 variables"
        #  assert np.allclose(data_test[9:], data_compare[9:], atol=1e-6), f"Arrays are not approximately equal for {filename} of index {idx} for last 4 variables"

        check_dirties(data_test, data_compare)

    assert 1 == 0, "All test cases passed"

def parser_total():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rate_undersampling', type=float, default=-1)
    parser.add_argument('--pos_weight', type=float, default=-1)
    parser.add_argument('--pos_step', type=int, default=0)
    parser.add_argument('--config', type=str, default="configs/feature_expert_full.yaml")
    parser.add_argument('--model', type=str, default="N/A")
    args = parser.parse_args()

    # Load config file
    config = None
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)

        print(f"Config:\n{config}")

    # Update arguments with values from the config file if they exist
    if config:
        for key, value in config.items():
            if hasattr(args, key) is False:
                setattr(args, key, value)
    else:
        raise ValueError(f"Invalid config file {args.config}")

    # Overwrite pos_weight if it's negative
    if args.pos_weight < 0:
        args.pos_weight = float(config.get('pos_weight', args.pos_weight))
    if args.rate_undersampling < 0:
        args.rate_undersampling = int(config.get('rate_undersampling', args.rate_undersampling))
    if args.model == "N/A":
        args.model = config.get('model', args.model)

    return args

def main():
    args = parser_total()

    print(args)
    # Timestamped unified experiment directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    experiment_dir = os.path.join("experiments", args.model,
                                  f"pos_{args.pos_step}",
                                  "rus_{:.2f}_pos_{:.2f}".format(args.rate_undersampling, args.pos_weight),
                                  f"run_{timestamp}")
    
    setattr(args, "exp_dir", experiment_dir)

    ckpt_dir = os.path.join(experiment_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    # Data
    data_module = get_data_module(args)

    print(len(data_module.train_set))
    print(len(data_module.val_set))
    print(len(data_module.test_set))
    print(len(data_module.train_dataloader()))
    print(len(data_module.val_dataloader()))
    print(len(data_module.test_dataloader()))
    
    print(data_module.train_set.df["Label"].value_counts())
    print(data_module.val_set.df["Label"].value_counts())
    print(data_module.test_set.df["Label"].value_counts())

    unit_testing_module(data_module)

if __name__ == '__main__':
    main()