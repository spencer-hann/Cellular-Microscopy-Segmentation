import tensorflow as tf
import numpy as np

import matplotlib.pyplot as plt

#from tqdm.keras import TqdmCallback

from ..data import data
from ..data.CustomImageDataGenerator import CustomImageDataGenerator
from ..Utils import plotter_wrapper, plot_save_name
from ..Utils.Options import using_options
from . import utils
from ..Training.ImageEntropyLoss import ImageEntropy

from logging import getLogger


log = getLogger(__name__)


@using_options
def train_model(
    model,
    train,
    val,
    viewers=None,
    opt=None,
    optimizer="adam",
    summarize=True,
    round=None,
):
    if optimizer == "adam":
        optimizer = tf.keras.optimizers.Adam(learning_rate=opt.lr)

    if viewers is None:
        viewers = CustomImageDataGenerator(name="view").flow_from_indices(
            [888, 2048],
            batch_size=2,
            make_null_mask=True,
        )

    if 1:
        log.info("Using binary crossentropy loss.")
        loss = tf.keras.losses.BinaryCrossentropy(
                from_logits=(opt.final_activation is None),)
    else:
        log.info("Using special sauce loss.")
        loss = ImageEntropy()

    log.info(f"Compiling model {opt.save_model_filename}")
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=[
            'accuracy',
            tf.keras.metrics.Precision(),
            tf.keras.metrics.Recall(),
            tf.keras.metrics.MeanIoU(num_classes=opt.output_channels),
        ]
    )

    display_callback = utils.MyDisplayCallback(
            model, viewers, n=4, round=round)
    checkpoint_callback = utils.make_checkpoint()

    if 1: display_callback.on_epoch_end()  # visualize
    if hasattr(model, "plot_model"):
        model.plot_model()
    log.debug('visualizations done')


    log.info(f'Beginning training {opt.steps_per_epoch} steps')
    model_history = model.fit(
        train, epochs=opt.epochs,
        steps_per_epoch=opt.steps_per_epoch,
        validation_steps=opt.val_steps_per_epoch,
        validation_data=val,
        #verbose=0,
        callbacks=(
            display_callback, checkpoint_callback, #TqdmCallback(verbose=2),
        ),
    )
    log.info('Done training')

    if summarize:
        summary(
            model_history,
            display_callback,
            save=plot_save_name(model, round, epoch="Summary"),
            round=round,
        )

    del display_callback.data
    del display_callback

    return model


@plotter_wrapper
@using_options
def summary(model_history, callback=None, opt=None, round="UNKNOWN"):
        log.info("Beginning Summary")
        if callback is not None:
            callback.on_epoch_end(epoch="final", round=round)  # visualize

        loss = model_history.history['loss']
        val_loss = model_history.history['val_loss']

        epoch_axis = range(opt.epochs)

        plt.figure()
        plt.plot(epoch_axis, loss, 'r', label='Training loss')
        plt.plot(epoch_axis, val_loss, 'bo', label='Validation loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss Value')
        plt.ylim([0, 1])
        plt.legend()

        log.info("End Summary")

