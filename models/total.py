from models.ResNet18 import ResNet18
from models.CNN2D import CNN2D
from models.CNN3D import CNN3D
from models.ViT import ViT_remaster as ViT

def get_model(args):
    if args.model == "ResNet":
        return ResNet18(in_channels=9, lr=args.lr, pos_weight=args.pos_weight), ResNet18
    elif args.model == "CNN2D":
        return CNN2D(in_channels=9, lr=args.lr, pos_weight=args.pos_weight), CNN2D
    elif args.model == "CNN3D":
        return CNN3D(in_channels=1, lr=args.lr, pos_weight=args.pos_weight), CNN3D
    elif args.model == "ViT":
        return ViT(lr=args.lr, pos_weight=args.pos_weight), ViT
    else:
        raise NotImplementedError