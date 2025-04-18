from data.feature_expert_full_map import (
    MerraFeatureExpertFullMapModule
)

from data.feature_expert_past import (
    MerraFeatureExpertPastModule
)

def get_data_module(args):
    print(f"args.batch_size: {args.batch_size}")
    print(f"args.rate_undersampling: {args.rate_undersampling}")
    print(f"args.agg_step: {args.agg_step}")
    if args.name == "FeatureExpertPast":
        return MerraFeatureExpertPastModule(
            batch_size=args.batch_size, rate_under_sampling=args.rate_undersampling,
            agg_step=args.agg_step
        )
    elif args.name == "FeatureExpertFull":
        return MerraFeatureExpertFullMapModule(
            folder_save=args.exp_dir,
            batch_size=args.batch_size,
            rate_under_sampling=args.rate_undersampling,
            agg_step=args.agg_step, pos_step=args.pos_step
        )
