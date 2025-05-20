import argparse
import yaml

def parser_total():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rate_undersampling', type=float, default=-1)
    parser.add_argument('--pos_weight', type=float, default=-1)
    parser.add_argument('--pos_step', type=int, default=-1)
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
    if args.pos_step < 0:
        args.pos_step = int(config.get('pos_step', args.pos_step))
    if args.model == "N/A":
        args.model = config.get('model', args.model)

    return args