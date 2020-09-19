import tensorflow as tf
import numpy as np

import time

from matplotlib import pyplot as plt
from itertools import islice, cycle
from IPython.display import clear_output
from logging import getLogger

from ..Training.PseudoLabelManager import PseudoLabelManager
from ..Utils import plotter_wrapper, iter_chunks, plot_save_name, make_stopwatch
from ..Utils.Options import using_options
from ..data import data
from ..data.CustomImageDataGenerator import CustomImageDataGenerator


log = getLogger(__name__)


def _subplot(coord, title, imgs, n, col, index=None):
    if imgs.shape[-1] < 3: imgs = imgs[...,0]
    else: imgs = imgs[...,:3]
    if index: index = iter(index)

    t = title
    for i, img in enumerate(imgs):
        plt.subplot(n, col, coord + (i*(col//2)))
        if index: t = title + ' ' + str(next(index))
        plt.title(t)
        plt.imshow(img, )


@plotter_wrapper
def visualize_predictions(data, model=None, n=6, indices=None,):
    plt.figure(figsize=(25, 5 * n))
    if model is None: col=4
    else: col=6

    if indices is None:
        indices = cycle(('','',))  # bunch'a empty strings
    indices = iter(indices)

    for i, (img, mask) in enumerate(islice(data, n)):
        _subplot(i*col+1, 'Input Image', img, n, col, index=next(indices))
        _subplot(i*col+2, 'True Mask', mask, n, col)
        if model is not None:
            yhat = model.predict(img)
            conf = PseudoLabelManager.confidence_matrix(yhat[...,:1])  # just cell mask
            conf = conf.max()
            t = f'Predicted Mask {yhat.min():.1f}, {yhat.mean():.2f}, {yhat.max():.2f}, {conf:.2f}'
            _subplot(i*col+3, t, yhat, n, col)


class MyDisplayCallback(tf.keras.callbacks.Callback):
    def __init__(self, model, data, n=6, round="UNKNOWN", save_every=1):
        self.model = model
        self.round = round
        self.save_every = save_every
        self.stopwatch = make_stopwatch()
        if data is None:
            log.warning("Display callback initialized without data.")
            self.data = None
        else:
            self.data = [(im, ma) for im, ma in islice(data, n)]
            if len(self.data) < n:
                log.error(
                    f"Display callback created with {len(self.data)} data (<{n})."
                )

    def on_epoch_end(self, epoch=None, logs=None, **kwargs):
        clear_output()
        tf.keras.backend.clear_session()  # release memory in use

        if isinstance(epoch, int):
            epoch += 1
        log.info(f"Epoch {epoch} end: {time.strftime('%X')}")
        log.info(f"Been training for {self.stopwatch()}")

        if (
            self.data is not None
            and (not isinstance(epoch, int) or epoch % self.save_every == 0)
        ):
            round = kwargs.pop("round", self.round)
            save = plot_save_name(self.model, round, epoch)
            visualize_predictions(
                self.data, self.model, len(self.data), save=save, **kwargs
            )


@using_options
def make_checkpoint(name=None, monitor='val_loss', opt=None):
    if name is None:
        name = opt.save_model_filename
    name = name + "e{epoch:02d}-l{val_loss:.2f}"
    path = opt.models_folder / name
    return tf.keras.callbacks.ModelCheckpoint(
        filepath=path,
        monitor=monitor,
        save_best_only=False,
        save_weights_only=False,
    )


def peruse_image_flow(
    indices=None,
    img_gen=None,
    model=None,
    input_shape=(512,512),
    n=2,
    default_dataset_step=100,
    downsample_step=1,
    reverse=False,
    make_null_mask=True,
    add_nuclei_to_cells=False,
    show=True,
    save=False,
    **kwargs,
):
    if input_shape == "raw":
        input_shape = data.raw_img_shape

    if indices is None:
        indices = np.arange(*data.cell_mask_range, default_dataset_step)
        indices = list(data.expell_test_indices(indices))

    if reverse:
        indices = indices[::-1]

    log.debug(indices)

    if img_gen is None:
        img_gen = CustomImageDataGenerator()

    testers = img_gen.flow_from_indices(
        indices.copy(),
        shape=input_shape,
        shuffle=False,
        batch_size=n,
        make_null_mask=make_null_mask,
        add_nuclei_to_cells=add_nuclei_to_cells,
    )

    indices = cycle(indices)
    indices = iter_chunks(indices, 2, wrap=tuple)
    indices = iter_chunks(indices, n, wrap=tuple)

    loopy = zip(indices, iter_chunks(testers, n, wrap=tuple))
    for ibatch, batch in loopy:
        print('making batch', flush=True)
        batch = map(lambda x: batch_setup(x, downsample_step), batch)
        print('done.', flush=True)

        log.debug(ibatch)
        print()
        visualize_predictions(
            batch,
            model=model,
            n=n,
            block=True,
            allow_interupt=False,
            indices=ibatch,
            show=show,
            save=save,
            **kwargs,
        )


def batch_setup(batch, downsample_step=1):
    if downsample_step != 1:  # lazy downsample
        img, mask = batch
        img = img[:, ::downsample_step, ::downsample_step, :]
        mask = mask[:, ::downsample_step, ::downsample_step, :]
        batch = img, mask

    return batch


def check_cell_similarity(
    indices=None,
    img_gen=None,
    input_shape=data.raw_img_shape,
    n=2,
    dataset_step=100,
    downsample_step=1,
    reverse=False,
):
    if indices is None:
        indices = np.arange(*data.cell_mask_range, dataset_step)

    if reverse:
        indices = indices[::-1]

    remainder = len(indices) % n
    if remainder:
        indices = indices[:-remainder]

    log.debug(indices)

    if img_gen is None:
        img_gen = CustomImageDataGenerator()

    testers = img_gen.flow_from_indices(
        indices.copy(),
        shape=input_shape,
        shuffle=False,
        batch_size=n,
        add_nuclei_to_cells=False,
    )

    ds = downsample_step

    for img, mask in testers:
        plt.tight_layout()

        img = img[:, ::ds, ::ds, 0]
        mask = mask[:, ::ds, ::ds, :]  # lazy downsample
        #mask = mask[..., 0]  # take cells
        mask = mask[-1]
        #mask = np.prod(mask, axis=0)
        #mask = np.stack((mask[0], mask[n//2], mask[-1]), axis=2)

        plt.subplot(131)
        plt.imshow(img[0])

        plt.subplot(132)
        plt.imshow(img[-1])

        plt.subplot(133)
        plt.imshow(mask)

        plt.show()

        plt.cla()

