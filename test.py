import argparse
import os
import yaml
import sys
from pytorch_lightning import Trainer

from data.total import get_data_module
from models.total import get_model
from evaluation.total import evaluate_model

# Custom Trainer that includes custom evaluation
class CustomTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def test(self, *args, **kwargs):
        super().test(*args, **kwargs)

        print("🔍 Running custom evaluation inside Trainer...")
        model = self.model if hasattr(self, "model") else args[0]
        datamodule = kwargs.get("datamodule", None)

        assert datamodule is not None, "⚠️ Custom evaluation skipped: No datamodule provided."

        test_loader = datamodule.test_dataloader()
        metrics_returned = evaluate_model(model, test_loader)

        for key, item in metrics_returned.items():
            print("{}: {:.5f}".format(key, item))

def parser_test():
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', type=str, required=True, help="Path to the experiment folder")
    args = parser.parse_args()

    
    # Load config from experiment folder
    config_path = os.path.join(args.folder, "configs.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
        print(f"Config:\n{config}")

    # Update arguments with values from the config file if they exist
    if config:
        for key, value in config.items():
            if hasattr(args, key) is False:
                setattr(args, key, value)
    else:
        raise ValueError(f"Invalid config file {args.config}")

    try:
        print(args.model)
    except:
        setattr(args, "model", args.folder.split("/")[-3])
        print(args.model)

    return args

def test_folder(args):
    sys.stdout = open(os.path.join(args.folder, "out_test.txt"), "w")
    sys.stderr = open(os.path.join(args.folder, "err_test.txt"), "w")
    
    # Data
    data_module = get_data_module(args)
    
    # Model
    model, model_func = get_model(args)
    
    # Load best checkpoint
    ckpt_dir = os.path.join(args.folder, "checkpoints")
    best_ckpt_path = os.path.join(ckpt_dir, sorted(os.listdir(ckpt_dir))[-1])
    print(f"🏆 Loading best model from {best_ckpt_path}")
    model = model_func.load_from_checkpoint(best_ckpt_path)
    
    # Test
    trainer = Trainer()
    trainer.test(model, datamodule=data_module)

    sys.stdout.close()
    sys.stderr.close()

def main():
    args = parser_test()

    test_folder(args)
    
    
if __name__ == '__main__':
    main()


settings = {
    "RUS 1:4 (CW dynamic)": [0.07,0.11,0.15,0.18,0.22,0.25,0.28,0.31,0.34],
    "RUS 1:10 (CW dynamic)": [0.06,0.11,0.14,0.18,0.21,0.24,0.28,0.29,0.33],
    "RUS 1:20 (CW dynamic)": [0.07,0.1,0.15,0.18,0.21,0.24,0.27,0.3,0.32],
    "RUS 1:30 (CW dynamic)": [0.07,0.11,0.14,0.18,0.22,0.24,0.28,0.3,0.33],
    "NO RUS (CW dynamic)": [0.06,0.11,0,14],

    "RUS 1:4 (CW balanced)": [],
    "RUS 1:10 (CW balanced)": [],

    "RUS 1:20 (CW balanced)": [],
    "RUS 1:30 (CW balanced)": [],
    "NO RUS (CW balanced)": []
}