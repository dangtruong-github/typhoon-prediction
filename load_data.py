##########################
## main.py (Full version with: best model loading, timestamped checkpoints, smart cleanup)
##########################
import argparse
import yaml

from data.total import get_data_module

def iterate_batch(loader):
    for batch in loader:
        pass

def parser_total():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rate_undersampling', type=float, default=2)
    parser.add_argument('--pos_weight', type=float, default=3)
    parser.add_argument('--pos_step', type=int, default=0)
    parser.add_argument('--config', type=str, default="configs/feature_expert_full.yaml")
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
        args.pos_weight = config.get('pos_weight', args.pos_weight)
    if args.rate_undersampling < 0:
        args.rate_undersampling = config.get('rate_undersampling', args.rate_undersampling)

    return args

def main():
    args = parser_total()

    print(args)

    # Data
    data_module = get_data_module(args)

    # print(len(data_module.train_set))
    # print(len(data_module.val_set))
    print(len(data_module.test_set))
    # print(len(data_module.train_dataloader()))
    # print(len(data_module.val_dataloader()))
    print(len(data_module.test_dataloader()))

    # iterate_batch(data_module.train_dataloader_random())
    # iterate_batch(data_module.val_dataloader())
    iterate_batch(data_module.test_dataloader())

if __name__ == '__main__':
    main()
