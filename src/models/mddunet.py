import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F

from .encoder import UNetEncoder
from .decoder import UNetDecoder, Type
from .losses import SoftDice, HardDice, simple_dice
from .layers import GradientReversalLayer

from torchmetrics.classification import MulticlassPrecision, MulticlassRecall

import monai

from .utils import wandb, predict
from enum import Enum

import copy
import optuna


class ForwardMode(Enum):
    train_adversarial = 1
    train_normal = 2
    predict = 3


class MDDUNet(pl.LightningModule):
    def __init__(self,
                 learning_rate,
                 learning_rate_layer_decay=1,
                 adversary_learning_rate=None,
                 input_channels=1,
                 base_num_features=32,
                 num_blocks_per_stage=2,
                 feature_mult=2,
                 num_stages=4,
                 mdd_depth=0,
                 props={
                        'norm_op_kwargs': {'eps': 1e-5, 'affine': True},
                        'dropout_op': None,
                        'dropout_op_kwargs': {'p': 0, 'inplace': True},
                        'conv_op_kwargs': {'stride': 1, 'dilation': 1, 'bias': False},
                       },
                 num_classes=3,
                 batch_size=10,
                 deep_supervision=False,
                 max_features=512,
                 dice_weight=0.5,
                 dice_smooth=1.,
                 weighted_ce=False,
                 validation_dims=3,
                 slice_dim=1,
                 targets=[],
                 use_scheduler=True,
                 nonlin='ReLU',
                 extensive_logging=True,
                 prune_below_dice=None,
                 xi=None,
                 freeze_encoder_stages=[],
                 freeze_decoder_stages=[]):

        super().__init__()
        """
        Params:
            ... to be documented
            slice_dim: dimension in volume which we slice. Does not include batch or channel dims!
            mdd_from_epoch: add mdd after some epochs. If none, MDD will not be applied.
        """

        self.learning_rate = learning_rate
        self.learning_rate_layer_decay = learning_rate_layer_decay
        assert learning_rate_layer_decay > 0

        self.validation_dims = validation_dims
        self.dice_weight = dice_weight
        self.targets = targets

        self.batch_size = batch_size

        self.extensive_logging = extensive_logging
        self.prune_below_dice = prune_below_dice

        if nonlin == 'ReLU':
            props['nonlin'] = nn.ReLU
            props['nonlin_kwargs'] = {'inplace': True}
        elif nonlin == 'LeakyReLU':
            props['nonlin'] = nn.LeakyReLU
            props['nonlin_kwargs'] = {'negative_slope': 1e-2, 'inplace': True}
        else:
            raise ValueError('`nonlin` is not valid')

        self.model = MDDUNetBase(input_channels, base_num_features, num_blocks_per_stage,
                                 feature_mult, num_stages, props, num_classes, deep_supervision,
                                 max_features, slice_dim=slice_dim, mdd_depth=mdd_depth)

        self.hard_dice = HardDice(average='micro', ignore_index=0)
        self.soft_dice = SoftDice(alpha=0.5, smooth=dice_smooth)
        self.mc_precision = MulticlassPrecision(num_classes=3, average='macro', ignore_index=0)
        self.mc_recall = MulticlassRecall(num_classes=3, average='macro', ignore_index=0)
        self.soft_dice_monai = monai.losses.DiceLoss(include_background=False, to_onehot_y=True)
        self.softmax = nn.Softmax(dim=1)
        self.slice_dim = slice_dim
        self.weighted_ce = weighted_ce
        self.use_scheduler = use_scheduler
        self.apply_mdd = False  # will be changed once adversary is instantiated
        self.stop_training = False
        self.xi = xi  # Stop training if `loss_src_adv > xi` in batch

        if adversary_learning_rate is None:
            self.adversary_learning_rate = self.learning_rate
        else:
            self.adversary_learning_rate = adversary_learning_rate

        self.freeze_params(freeze_encoder_stages, freeze_decoder_stages)
        self.save_hyperparameters()

    def freeze_params(self, encoder_stages, decoder_stages):
        if len(encoder_stages + decoder_stages) == 0:
            return

        assert type(encoder_stages[0]) is int if len(encoder_stages) > 0 else True
        assert type(decoder_stages[0]) is int if len(decoder_stages) > 0 else True

        prefixes = [f"model.encoder.stages.{i}" for i in encoder_stages] + \
                   [f"model.decoder.stages.{i}" for i in decoder_stages] + \
                   [f"model.decoder.tus.{i}" for i in decoder_stages]

        for name, parameter in self.named_parameters():
            for prefix in prefixes:
                if prefix in name:
                    parameter.requires_grad = False

    def add_adversary(self, alpha, gamma, scaled_grl, max_iter, loss_reduction='mean'):
        if self.xi is None:
            print("Applying MDD without early stopping...")
        else:
            print(f"Applying MDD with early stopping at the xi={self.xi} level")

        self.apply_mdd = True
        self.mdd_loss_reduction = loss_reduction
        self.model.add_adversary(alpha, gamma, scaled_grl, max_iter)
        wandb.update_config({
            'alpha': alpha,
            'gamma': gamma,
            'scaled_grl': scaled_grl,
            'max_iter': max_iter,
            'loss_reduction': loss_reduction
        })

    def on_train_batch_start(self, batch, batch_idx):
        if self.stop_training:
            print("Training stopped. Skipping epoch...")
            return -1
        else:
            return 1

    def training_step(self, batch, batch_idx):
        xs, ys = batch['source']
        xt, _ = batch['target']

        assert len(xs.shape) == 4
        assert len(xt.shape) == 4
        assert len(ys.shape) == 3

        weights = None
        if self.weighted_ce:
            weights = self.trainer.datamodule.ce_weights
            weights = weights.type_as(xs)

        if self.apply_mdd:
            logits_src_clf, logits_src_adv = self.model(xs, mode=ForwardMode.train_adversarial)
            logits_tgt_clf, logits_tgt_adv = self.model(xt, mode=ForwardMode.train_adversarial)

            if self.current_epoch % 5 == 0 and batch_idx == 0 and self.extensive_logging:
                wandb.plot_batch_logits(
                    xs.squeeze(), xt.squeeze(), logits_src_clf, logits_src_adv, logits_tgt_clf,
                    logits_tgt_adv, self.logger, self.slice_dim
                )

            loss_combined, loss_dict = self.model.loss(
                logits_src_clf, logits_src_adv, logits_tgt_clf, logits_tgt_adv, ys, weights=weights,
                reduction=self.mdd_loss_reduction
            )

            loss_src_adv = loss_dict["train/loss/adversary/loss_src_adv"]
            if self.xi is not None and loss_src_adv > self.xi:
                print(f"Stopping early! loss_src_adv: {loss_src_adv}, xi: {self.xi}")
                self.stop_training = True
        else:
            cross_entropy = nn.CrossEntropyLoss(weight=weights)
            logits_src_clf, _ = self.model(xs, mode=ForwardMode.train_normal)
            loss_combined = cross_entropy(logits_src_clf, ys)
            loss_dict = {
                "train/loss/combined": loss_combined.data.item(),
                "train/loss/classifier": loss_combined.data.item(),
            }

        # Log training metrics
        p_src_clf = self.softmax(logits_src_clf)
        ys_hat = torch.argmax(p_src_clf, dim=1)

        if batch_idx % 100 == 0 and self.extensive_logging:
            wandb.log_image_2d(xs, ys_hat, ys, self.logger, source="train")

        soft_dice = 1 - self.soft_dice(p_src_clf, ys.unsqueeze(1))
        soft_dice_monai = self.soft_dice_monai(p_src_clf, ys.unsqueeze(1))
        hard_dice = self.hard_dice(ys_hat, ys)
        simple_hard_dice, precision, recall = simple_dice(ys_hat, ys)

        loss_dict["train/eval/soft_dice"] = soft_dice.data.item()
        loss_dict["train/eval/soft_dice_monai"] = soft_dice_monai.data.item()
        loss_dict["train/eval/hard_dice"] = hard_dice.data.item()
        loss_dict["train/eval/simple_hard_dice"] = simple_hard_dice
        loss_dict["train/eval/precision"] = precision
        loss_dict["train/eval/recall"] = recall

        self.log_dict(loss_dict, batch_size=self.batch_size)

        assert not loss_combined.isnan().any()
        return loss_combined

    def validation_step(self, batch, batch_idx):
        assert self.validation_dims == 3
        x, y = batch  # both (B, H, W, D) where B = 1

        assert x.shape[0] == 1
        assert y.shape[0] == 1
        y = y.squeeze()  # (H, W, D)

        y_hat = predict.predict3d(x, self.model.predict, self.slice_dim)

        assert y_hat.shape == y.shape, (y_hat.shape, y.shape)

        dice = self.hard_dice(y_hat, y)
        precision = self.mc_precision(y_hat, y)
        recall = self.mc_recall(y_hat, y)
        dice_val = dice.data.item()

        simple_hard_dice, simple_precision, simple_recall = simple_dice(y_hat, y)

        nonzero_preds = torch.nonzero(y_hat).cpu()
        nonzero_count = float(nonzero_preds.size(0))

        self.log_dict({
            "val/dice": dice.data.item(),
            "val/precision": precision.data.item(),
            "val/recall": recall.data.item(),
            "val/simple_dice": simple_hard_dice,
            "val/simple_precision": simple_precision,
            "val/simple_recall": simple_recall,
            "val/preds/nonzero": nonzero_count
        })

        if self.prune_below_dice is not None:
            if dice_val < self.prune_below_dice:
                wandb.finish()
                raise optuna.TrialPruned(f"Pruned at {dice_val}")

        if batch_idx == 5 and self.extensive_logging:
            wandb.log_image_3d(
                x, y_hat, y, self.logger, slice_dim=self.slice_dim, source="val", text=dice_val
            )

        # log the predicts for debugging
        if self.current_epoch % 15 == 0 and self.extensive_logging:
            if batch_idx == 5:
                wandb.save_tensor(y, "y.pkl")
                wandb.save_tensor(y_hat, "y_hat.pkl")

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch  # both (B, H, W, D) where B = 1

        assert x.shape[0] == 1
        assert y.shape[0] == 1
        y = y.squeeze()  # (H, W, D)

        y_hat = predict.predict3d(x, self.model.predict, self.slice_dim)
        dice = self.hard_dice(y_hat, y)
        precision = self.mc_precision(y_hat, y)
        recall = self.mc_recall(y_hat, y)

        simple_hard_dice, simple_precision, simple_recall = simple_dice(y_hat, y)

        dataset = self.targets[dataloader_idx]
        self.log_dict({
            f"test/{dataset}/dice": dice.data.item(),
            f"test/{dataset}/precision": precision.data.item(),
            f"test/{dataset}/recall": recall.data.item(),
            f"test/{dataset}/simple_dice": simple_hard_dice,
            f"test/{dataset}/simple_precision": simple_precision,
            f"test/{dataset}/simple_recall": simple_recall,
        })

        if batch_idx == 3:
            wandb.log_image_3d(
                x, y_hat, y,
                self.logger,
                slice_dim=self.slice_dim,
                source=f"test/{dataset}",
                text=dice
            )

    def configure_optimizers(self):
        lr = self.learning_rate
        decay = self.learning_rate_layer_decay

        parameter_groups = [
            {
                'name': 'encoder',
                'params': filter(lambda p: p.requires_grad, self.model.encoder.parameters()),
                'lr': lr/(decay**2)
            },
            {
                'name': 'decoder',
                'params': filter(lambda p: p.requires_grad, self.model.decoder.parameters()),
                'lr': lr/decay
            },
            {
                'name': 'classifier',
                'params': self.model.classifier.parameters(),
                'lr': lr
            },
        ]

        if self.apply_mdd:
            parameter_groups.append({
                'params': self.model.adversary.parameters(),
                'name': 'adversary',
                'lr': self.adversary_learning_rate
            })

        optimizer = torch.optim.Adam(parameter_groups, lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=.5, verbose=True)

        if self.use_scheduler:
            return [optimizer], [scheduler]
        return optimizer


class MDDUNetBase(nn.Module):
    def __init__(self,
                 input_channels,
                 base_num_features,
                 num_blocks_per_stage,
                 feature_mult,
                 num_stages,
                 props,
                 num_classes,
                 deep_supervision,
                 max_features,
                 slice_dim=1,
                 gamma=1.,
                 mdd_depth=0,
                 scaled_grl=False,
                 alpha=1.,):
        super().__init__()

        self.num_classes = num_classes
        self.slice_dim = slice_dim

        # softmax dims: (B, C, H, W), and we wish to softmax over the class dimensioin
        self.softmax = nn.Softmax(dim=1)

        self.encoder = UNetEncoder(input_channels, base_num_features, num_blocks_per_stage,
                                   feature_mult, num_stages, props, max_num_features=max_features)
        assert mdd_depth <= num_stages
        self.mdd_depth = mdd_depth
        # 0: only segmentation layer is adversarily trained
        # 1: "top" stage is adversary
        # 2: top two stages are adversaty...

        # Non-adverserial part of decoder
        self.decoder = UNetDecoder(self.encoder, num_classes, type=Type.bottom, depth=mdd_depth)

        # min-player
        self.classifier = UNetDecoder(self.encoder, num_classes, type=Type.head, depth=mdd_depth)

    def add_adversary(self, alpha, gamma, scaled_grl, max_iter):
        # max-player
        self.adversary = copy.deepcopy(self.classifier)

        self.grl = GradientReversalLayer(
           alpha=alpha, high=1., low=0., max_iter=max_iter, scale=scaled_grl
        )

        self.gamma = gamma

    def forward(self, x, mode=ForwardMode.train_adversarial):
        assert len(x.shape) == 4
        assert x.shape[1] == 1  # 1 channel
        assert mode in list(ForwardMode)

        skips = self.encoder(x)
        # split skips into bottleneck, top and bottom
        bottleneck = skips[-1]
        top_skips = skips[:self.mdd_depth] if self.mdd_depth > 0 else []
        bottom_skips = skips[self.mdd_depth:-1] if self.mdd_depth > 0 else skips[:-1]
        assert len(bottom_skips) + len(top_skips) + 1 == len(skips)

        features = self.decoder(bottleneck, bottom_skips)

        logits_clf = self.classifier(features, top_skips)
        logits_adv = None

        if mode == ForwardMode.train_adversarial:
            features_adv = self.grl(features)
            skips_adv = self.grl(top_skips)  # TODO: Verify
            logits_adv = self.adversary(features_adv, skips_adv)

        return logits_clf, logits_adv

    def predict(self, x):
        assert len(x.shape) == 4
        logits, _ = self.forward(x, mode=ForwardMode.predict)
        p = self.softmax(logits)       # (B, num_classes, H, W)
        return torch.argmax(p, dim=1)  # arg_max over the num_classes dim so output is (B, H, W)

    def loss(self, logits_src_clf, logits_src_adv, logits_tgt_clf, logits_tgt_adv, y_src,
             weights=None, reduction='mean'):
        """
            `reduction` should either be 'mean' or 'none' and specifies whether we mean-reduce
            pixels when computing the cross entropy or after computing the mdd loss.
        """
        eps = 1e-8
        cross_entropy = nn.CrossEntropyLoss(weight=weights, reduction=reduction)

        y_src_clf = logits_src_clf.max(1)[1]
        y_tgt_clf = logits_tgt_clf.max(1)[1]

        # eq (28)
        loss_clf = cross_entropy(logits_src_clf, y_src)
        loss_src_adv = cross_entropy(logits_src_adv, y_src_clf)

        # eq (29)
        p_tgt_adv = F.softmax(logits_tgt_adv, dim=1)
        logloss_tgt = torch.log(1 - p_tgt_adv + eps)
        loss_tgt_adv = F.nll_loss(logloss_tgt, y_tgt_clf)

        if reduction == 'none':
            loss_clf = loss_clf.mean(axis=0)  # average over the batch dimension, so dim = (H, W)
            loss_src_adv = loss_src_adv.mean(axis=0)
            loss_tgt_adv = loss_tgt_adv.mean(axis=0)

        # eq (30)
        loss_transfer = (self.gamma * loss_src_adv) + loss_tgt_adv

        # final loss
        loss_combined = loss_clf + loss_transfer

        if reduction == 'none':
            loss_combined = loss_combined.mean()
            loss_clf = loss_clf.mean()
            loss_transfer = loss_transfer.mean()
            loss_src_adv = loss_src_adv.mean()
            loss_tgt_adv = loss_tgt_adv.mean()

        loss_dict = {
            "train/loss/combined": loss_combined.data.item(),
            "train/loss/classifier": loss_clf.data.item(),
            "train/loss/adversary": loss_transfer.data.item(),
            "train/loss/adversary/loss_src_adv": loss_src_adv.data.item(),
            "train/loss/adversary/loss_tgt_adv": loss_tgt_adv.data.item(),
            "train/loss/adversary/max_p_tgt_adv": torch.max(p_tgt_adv).data.item()
        }

        return loss_combined, loss_dict
