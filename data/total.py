from data.merra2_loader import (
    Merra2_fullmap_Loader,
    Merra2_past_Loader
)

def get_data_module(args):
    print(f"args.batch_size: {args.batch_size}")
    print(f"args.rate_undersampling: {args.rate_undersampling}")
    print(f"args.agg_step: {args.agg_step}")
    
    if args.type_data == "fullmap":
        return Merra2_fullmap_Loader(
            folder_save=args.exp_dir,
            batch_size=args.batch_size,
            rate_under_sampling=args.rate_undersampling,
            agg_step=args.agg_step, pos_step=args.pos_step,
            type_retrieve=args.type_retrieve
        )
    elif args.type_data == "past":
        return Merra2_past_Loader(
            folder_save=args.exp_dir,
            batch_size=args.batch_size,
            rate_under_sampling=args.rate_undersampling,
            agg_step=args.agg_step, pos_step=args.pos_step,
            type_retrieve=args.type_retrieve
        )
