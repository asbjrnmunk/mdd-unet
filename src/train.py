import argparse
import torch
import pytorch_lightning as pl

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor

import wandb
from wandb import AlertLevel

from models import MDDUNet  # , UNet
from datasets import MRIDataModule

SEED = 1201912496


def get_non_default_params(parser, args):
    options = [
        opt for opt in parser._option_string_actions.values() if hasattr(args, opt.dest)
    ]
    non_default = {}

    for option in options:
        val = getattr(args, option.dest)
        if option.default != val:
            non_default[option.dest] = val

    return non_default


def boolean_string(s):
    if s not in {"False", "True"}:
        raise ValueError("Not a valid boolean string")
    return s == "True"


if __name__ == "__main__":
    print("CUDA VERSION USED: ", torch.version.cuda)
    print("CUDA IS AVAILABLE: ", torch.cuda.is_available())

    pl.seed_everything(SEED)
    parser = argparse.ArgumentParser()

    # slurm config
    parser.add_argument(
        "--require_gpu", action="store_true", help="fail if no gpu available"
    )

    # training config
    parser.add_argument("--experiment", type=str, default="")
    parser.add_argument("--source", type=str, default="hammers")
    parser.add_argument("--validation", type=str, default="lpba40")
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

    # mdd config
    parser.add_argument("--scaled_grl", type=boolean_string, default=True)
    parser.add_argument("--mdd_depth", type=int, default=0)
    parser.add_argument("--alpha", type=int, default=1.4)
    parser.add_argument("--gamma", type=float, default=0.08)
    parser.add_argument("--adversary_learning_rate", type=float, default=None)
    parser.add_argument("--learning_rate_layer_decay", type=float, default=1.5)
    parser.add_argument(
        "--mdd_on", type=str, default=None, help="Checkpoint to train mdd on"
    )
    parser.add_argument("--mdd_loss_reduction", type=str, default="mean")

    parser.add_argument(
        "--xi", type=float, default=0.02, help="Stop epoch when xi > L^s_a"
    )
    parser.add_argument("--no_early_stopping", action="store_true")

    # freeze params
    parser.add_argument(
        "--freeze_encoder_stages", nargs="*", help="0=top", default=[], type=int
    )
    parser.add_argument(
        "--freeze_decoder_stages", nargs="*", help="0=bottom", default=[], type=int
    )

    args = parser.parse_args()

    offline = (not torch.cuda.is_available()) and (args.mdd_on is None)

    if args.require_gpu:
        assert torch.cuda.is_available()

    datasets = ["hammers", "lpba40", "oasis", "harp"]

    # check if we are doing MDD
    if args.mdd_on is not None:
        targets = [args.validation]
        net = "MDDUNet"
    elif args.validation.endswith("_test"):
        targets = [args.validation]
        net = "UNet"
    else:
        targets = datasets
        net = "UNet"

    tags = [f"s:{args.source}", f"v:{args.validation}", f"t:{'_'.join(targets)}"]

    experiment = str(get_non_default_params(parser, args)).strip("{}")

    if args.source == args.validation:
        run_name = f"Debug run {experiment}"
        notes = "Debug"
        tags.append("debug")
    else:
        run_name = f"{net} {experiment}"
        notes = "DA"

    print(f"Running {run_name} with {args}")

    wandb_logger = WandbLogger(
        name=run_name,
        notes=notes,
        tags=tags,
        project="mdd-unet",
        offline=offline,
        log_model=not offline,
    )

    wandb.alert(title="Job started", text=f"Job: {run_name}", level=AlertLevel.INFO)

    if args.mdd_on:
        if args.mdd_on.endswith(".ckpt"):
            ckpt = args.mdd_on
        else:
            artifact = wandb_logger.use_artifact(f"model-{args.mdd_on}:latest")
            path = artifact.download()
            ckpt = f"{path}/model.ckpt"

        model = MDDUNet.load_from_checkpoint(
            ckpt,
            learning_rate=args.learning_rate,
            adversary_learning_rate=args.adversary_learning_rate,
            learning_rate_layer_decay=args.learning_rate_layer_decay,
            targets=targets,
            use_scheduler=not args.no_scheduler,
            xi=args.xi if not args.no_early_stopping else None,
            freeze_encoder_stages=args.freeze_encoder_stages,
            freeze_decoder_stages=args.freeze_decoder_stages,
            batch_size=args.batch_size,
        )
        model.add_adversary(
            alpha=args.alpha,
            gamma=args.gamma,
            scaled_grl=args.scaled_grl,
            max_iter=args.max_iter,
            loss_reduction=args.mdd_loss_reduction,
        )
    else:
        model = MDDUNet(
            learning_rate=args.learning_rate,
            num_stages=args.num_stages,
            base_num_features=args.base_num_features,
            num_blocks_per_stage=args.num_blocks_per_stage,
            feature_mult=args.feature_mult,
            mdd_depth=args.mdd_depth,
            dice_weight=args.dice_weight,
            weighted_ce=args.weighted_ce,
            validation_dims=args.validation_dims,
            slice_dim=args.slice_dim,
            targets=targets,
            use_scheduler=not args.no_scheduler,
            nonlin=args.nonlin,
            freeze_encoder_stages=args.freeze_encoder_stages,
            freeze_decoder_stages=args.freeze_decoder_stages,
            batch_size=args.batch_size,
        )

    if len(args.freeze_encoder_stages + args.freeze_decoder_stages) > 0:
        for name, params in model.named_parameters():
            print(
                f"{'name: ' + name:<50}"
                f"{'size: ' + str(list(params.shape)):<25}"
                f"{'requires_grad: ' + str(params.requires_grad)}"
            )

    wandb_logger.experiment.config.update(
        {
            "prob_fg": args.prob_fg,
            "epoch_size": args.epoch_size if args.use_augmentations else None,
            "ignore_empty": not args.consider_empty,
            "device": "cpu"
            if not torch.cuda.is_available()
            else torch.cuda.get_device_name(0),
            "gradient_clip_value": args.gradient_clip_val,
            "gradient_clip_algorithm": args.gradient_clip_algorithm,
            "slice_dim": model.slice_dim,
            "freeze_encoder_stages": args.freeze_encoder_stages,
            "freeze_decoder_stages": args.freeze_decoder_stages,
        }
    )

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

    lr_monitor = LearningRateMonitor(logging_interval="epoch", log_momentum=True)

    trainer = pl.Trainer(
        callbacks=[lr_monitor],
        max_epochs=args.max_epochs,
        accelerator="auto",
        devices=1,
        check_val_every_n_epoch=1,
        logger=wandb_logger,
        log_every_n_steps=25,
        gradient_clip_val=args.gradient_clip_val,
        gradient_clip_algorithm=args.gradient_clip_algorithm,
        track_grad_norm=args.track_grad_norm,
        profiler="simple",
    )

    wandb_logger.watch(model, log_freq=args.wandb_model_log_freq)

    trainer.fit(model, datamodule=data)

    if len(targets) > 0:
        trainer.test(model, datamodule=data)
