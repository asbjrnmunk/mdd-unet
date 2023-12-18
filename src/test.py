import argparse
import torch
import pytorch_lightning as pl

from pytorch_lightning.loggers import WandbLogger

# import wandb

from models import MDDUNet  # , UNet
from datasets import MRIDataModule

SEED = 1201912496

if __name__ == "__main__":
    print("CUDA VERSION USED: ", torch.version.cuda)
    print("CUDA IS AVAILABLE: ", torch.cuda.is_available())

    pl.seed_everything(SEED)
    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoint", type=str)

    parser.add_argument(
        "--require_gpu", action="store_true", help="fail if no gpu available"
    )

    parser.add_argument("--source", type=str, default="hammers")
    parser.add_argument("--validation", type=str, default="alle")
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument(
        "--epoch_size", type=int, default=10000, help="Only used with --use_aug.."
    )
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument(
        "--prob_fg", type=int, default=None, help="use None for dataset value"
    )
    parser.add_argument("--consider_empty", action="store_true")
    parser.add_argument("--max_epochs", type=int, default=60)
    parser.add_argument("--slice_dim", type=int, default=1)
    parser.add_argument("--force_preprocess", action="store_true")
    parser.add_argument(
        "--gradient_clip_val", type=float, default=0.5, help="use `0` to turn off"
    )
    parser.add_argument("--gradient_clip_algorithm", type=str, default="norm")
    parser.add_argument("--max_iter", type=int, default=60 * 1040)
    parser.add_argument("--track_grad_norm", type=int, default=2)
    parser.add_argument("--wandb_model_log_freq", type=int, default=100)

    # model config
    parser.add_argument("--num_stages", type=int, default=4)
    parser.add_argument("--base_num_features", type=int, default=32)
    parser.add_argument("--num_blocks_per_stage", type=int, default=2)
    parser.add_argument("--feature_mult", type=int, default=2)
    parser.add_argument("--dice_weight", type=int, default=0)
    parser.add_argument("--weighted_ce", action="store_true")
    parser.add_argument("--validation_dims", type=int, default=3)
    parser.add_argument("--no_scheduler", action="store_true")
    parser.add_argument(
        "--nonlin", type=str, default="ReLU", choices=["ReLU", "LeakyReLU"]
    )

    parser.add_argument("--use_augmentations", action="store_true")
    parser.add_argument("--p_noise_aug", type=float, default=0.25)
    parser.add_argument("--p_gamma_aug", type=float, default=0.3)
    parser.add_argument("--p_rot_aug", type=float, default=0.25)
    parser.add_argument("--p_blur_aug", type=float, default=0.2)
    parser.add_argument("--p_elastic_aug", type=float, default=0)

    args = parser.parse_args()

    offline = False  # not torch.cuda.is_available()

    if args.require_gpu:
        assert torch.cuda.is_available()

    if args.validation == "alle":
        targets = ["hammers", "lpba40", "oasis", "harp"]
    else:
        targets = [args.validation]

    wandb_logger = WandbLogger(
        name=f"{args.checkpoint} source: {args.source}",
        project="mdd-unet",
        offline=offline,
        log_model=not offline,
    )

    artifact = wandb_logger.use_artifact(f"model-{args.checkpoint}:latest")
    path = artifact.download()
    ckpt = f"{path}/model.ckpt"

    model = MDDUNet.load_from_checkpoint(ckpt)

    data = MRIDataModule(
        source=args.source,
        val=args.validation,
        targets=targets,
        epoch_size=args.epoch_size if args.use_augmentations else None,
        slice_dim=model.slice_dim,
        prob_fg=args.prob_fg,
        ignore_empty=not args.consider_empty,
        use_augmentations=args.use_augmentations,
        p_blur_aug=args.p_blur_aug,
        p_noise_aug=args.p_noise_aug,
        p_gamma_aug=args.p_gamma_aug,
        p_rot_aug=args.p_rot_aug,
        p_elastic_aug=args.p_elastic_aug,
        force_preprocess=args.force_preprocess,
    )

    trainer = pl.Trainer(logger=wandb_logger, resume_from_checkpoint=ckpt)
    trainer.test(model, datamodule=data)
