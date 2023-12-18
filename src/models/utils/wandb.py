import torch
from torch.nn import functional as F

import wandb

from pytorch_lightning.loggers import WandbLogger

import matplotlib.pyplot as plt
import numpy as np


def finish():
    wandb.finish()


def update_config(dct):
    wandb.config.update(dct)


def save_tensor(x, name):
    torch.save(x, name)
    wandb.save(name)
    print(f"Saved {name} to wandb")


def soft_argmax(logits, dim=1):
    p = F.softmax(logits, dim=dim)
    return torch.argmax(p, dim=dim)


def plot_batch_logits(xs, xt, logits_src_clf, logits_src_adv, logits_tgt_clf, logits_tgt_adv,
                      logger, slice_dim):
    xs = xs.detach().cpu()
    xt = xt.detach().cpu()
    y_src_clf = soft_argmax(logits_src_clf.detach(), dim=slice_dim).cpu()
    y_src_adv = soft_argmax(logits_src_adv.detach(), dim=slice_dim).cpu()
    y_tgt_clf = soft_argmax(logits_tgt_clf.detach(), dim=slice_dim).cpu()
    y_tgt_adv = soft_argmax(logits_tgt_adv.detach(), dim=slice_dim).cpu()

    pngs = []

    for i in range(xs.shape[0]):
        png = plot_logits(
            xs[i, :, :],
            xt[i, :, :],
            y_src_clf[i, :, :],
            y_src_adv[i, :, :],
            y_tgt_clf[i, :, :],
            y_tgt_adv[i, :, :]
        )

        pngs.append(png)

    logger.experiment.log({"train/examples/logits": pngs})

    plt.close('all')


def plot_logits(xsi, xti, yi_src_clf, yi_src_adv, yi_tgt_clf, yi_tgt_adv):
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8, 8))

    colors = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    cmap = plt.matplotlib.colors.ListedColormap(colors)

    axes[0, 0].imshow(xsi, cmap='gray')
    axes[0, 0].imshow(yi_src_clf, cmap=cmap, alpha=0.5)
    axes[0, 0].set_title('logits_src_clf')

    axes[0, 1].imshow(xsi, cmap='gray')
    axes[0, 1].imshow(yi_src_adv, cmap=cmap, alpha=0.5)
    axes[0, 1].set_title('logits_src_adv')

    axes[1, 0].imshow(xti, cmap='gray')
    axes[1, 0].imshow(yi_tgt_clf, cmap=cmap, alpha=0.5)
    axes[1, 0].set_title('logits_tgt_clf')

    axes[1, 1].imshow(xti, cmap='gray')
    axes[1, 1].imshow(yi_tgt_adv, cmap=cmap, alpha=0.5)
    axes[1, 1].set_title('logits_tgt_adv')

    sm = plt.cm.ScalarMappable(cmap=cmap)
    sm.set_array(yi_src_clf)

    cbar = plt.colorbar(sm, ax=axes.ravel().tolist(),
                        location='bottom',
                        shrink=0.5,
                        pad=0.05)
    cbar.set_label('Segmentation')

    return wandb.Image(fig)


def log_image_3d(x, y_hat, y, logger, source, slice_dim, text=None):
    assert len(x.shape) == 4
    assert len(y_hat.shape) == 3, y_hat.shape
    assert len(y.shape) == 3, y.shape
    assert y.shape == y_hat.shape

    # x: (1, H, X, Y)
    # y_hat: (H, X, Y)
    # y: (H, X, Y)

    x = x.squeeze().detach().cpu()
    y_hat = y_hat.detach().cpu()
    y = y.detach().cpu()

    imgs = []

    for i in torch.arange(x.shape[slice_dim]):
        xi = torch.index_select(x, slice_dim, i).squeeze()

        if xi.min() == xi.max():
            continue  # slice is empty

        yi_hat = torch.index_select(y_hat, slice_dim, i).squeeze()
        yi = torch.index_select(y, slice_dim, i).squeeze()

        if yi.min() == yi.max() and yi_hat.min() == yi_hat.max():
            continue  # we have neither foreground in pred or gt

        img = create_image(xi, yi_hat, yi, text)
        imgs.append(img)

    assert len(imgs) > 0

    logger.experiment.log({f"{source}/examples": imgs})
    plt.close('all')


def log_image_2d(x, y_hat, y, logger, source="train"):
    assert isinstance(logger, WandbLogger)

    # x: (B, 1, X, Y)
    # y_hat: (B, X, Y)
    # y: (B, X, Y)

    assert len(x.shape) == 4
    assert len(y_hat.shape) == 3
    assert len(y.shape) == 3

    y_hat_fg_idx = torch.unique(torch.nonzero(y_hat)[:, 0])
    if len(y_hat_fg_idx) > 0:
        idx = y_hat_fg_idx[len(y_hat_fg_idx) // 2]
    else:
        y_fg_idx = torch.unique(torch.nonzero(y)[:, 0])
        if len(y_fg_idx) == 0:
            return
        idx = y_fg_idx[len(y_fg_idx) // 2]

    xi = x[idx].cpu().squeeze()
    yi_hat = y_hat[idx].cpu()
    yi = y[idx].cpu()

    img = create_image(xi, yi_hat, yi)
    logger.experiment.log({f"{source}/examples": [img]})
    plt.close('all')


def create_image(xi, yi_hat, yi, text=None):
    # assumes inputs have dims
    # x: (X, Y)
    # y_hat: (X, Y)
    # y: (X, Y)

    assert len(xi.shape) == 2, xi.shape
    assert len(yi_hat.shape) == 2, yi_hat.shape
    assert len(yi.shape) == 2, yi.shape

    assert yi_hat.shape == yi.shape, (yi_hat.shape, yi.shape)

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))

    colors = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    cmap = plt.matplotlib.colors.ListedColormap(colors)

    axes[0].imshow(xi, cmap='gray')
    axes[0].imshow(yi, cmap=cmap, alpha=0.5)
    axes[0].set_title('Ground Truth')

    axes[1].imshow(xi, cmap='gray')
    axes[1].imshow(yi_hat, cmap=cmap, alpha=0.5)
    axes[1].set_title(f"Prediction (dice={text})")

    sm = plt.cm.ScalarMappable(cmap=cmap)
    sm.set_array(yi)

    cbar = plt.colorbar(sm, ax=axes.ravel().tolist(),
                        location='bottom',
                        shrink=0.5,
                        pad=0.05)
    cbar.set_label('Segmentation')

    return wandb.Image(fig)
