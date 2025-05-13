from models.ResNet18 import ResNet18
from models.CNN2D import CNN2D
from models.CNN3D import CNN3D
from models.ViT import ViT_remaster as ViT
from models.TCGPNet import TCGPNet

from models.BaseModel import BaseModel

def get_model(args):
    model = retrieve_model(args)
    model_trainer = BaseModel(model, lr=args.lr, pos_weight=args.pos_weight)

    return model_trainer
    
def retrieve_model(args):
    if args.model == "ResNet":
        return ResNet18(in_channels=13)
    elif args.model == "CNN2D":
        return CNN2D(in_channels=13)
    elif args.model == "CNN3D":
        return CNN3D(in_channels=1)
    elif args.model == "ViT":
        return ViT(lr=args.lr)
    elif args.model == "SwinTransformer":  
        predictor_configs = [1, 1, 1, 25, 25, 25, 25, 25, 25, 25, 25, 25]
        return TCGPNet(
            img_size = (33, 33),
            predictor_configs=predictor_configs,
            patch_size=4,
            window_size=4,
            depths=[2, 2],
            num_heads=[3, 6],
            num_classes=1
        )
    else:
        raise NotImplementedError