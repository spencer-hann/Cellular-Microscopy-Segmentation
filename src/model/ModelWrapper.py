import tensorflow as tf
import numpy as np

import logging

from ..data import data
from ..Utils import Options
from ..data.CustomImageDataGenerator import CustomImageDataGenerator


log = logging.getLogger(__name__)


def inrange_assert(x):
    assert (x >= 0.), (x.min(), x.mean(), x.max())
    assert (x <= 1.), (x.min(), x.mean(), x.max())


class ModelWrapper:
    @Options.using_options
    def __init__(
        self,
        models,
        input_shape=None,
        output_channels=None,
        model_weights=None,
        opt=None,
        trainable=False,
    ):
        if input_shape is None:
            input_shape = opt.input_shape
        self.input_shape = input_shape

        if output_channels is None:
            output_channels = 3
        self.output_channels = output_channels

        if isinstance(models, tf.keras.models.Model):
            models = (models,)
        self.models = models

        for m in models:
            m.trainable = trainable

        if model_weights is None:
            n = len(models)
            model_weights = [1/n] * n

        model_weights = np.array(model_weights)

        self.model_weights = model_weights

    @staticmethod
    def _predict_crop(model, img, flip=True):
        out = model.predict(img)
        if flip:
            out += model.predict(img[:,    :, ::-1, :])
            out += model.predict(img[:, ::-1,    :, :])
            out += model.predict(img[:, ::-1, ::-1, :])
            out /= 4.

        inrange_assert(out)
        return out

    def predict_crop(self, img, flip=True):
        if self.input_shape != img.shape[1:3]:
            raise RuntimeError("shape mismatch")

        outshape = self.make_outputshape(img)
        yhat = np.zeros(outshape, dtype=img.dtype)

        for m, w in zip(self.models, self.model_weights):
            _yhat = self._predict_crop(m, img, flip=False)
            inrange_assert(_yhat)
            yhat += w * _yhat
            inrange_assert(yhat)

        return yhat

    def predict(self, img, overlap=0.5, prediction_threshold=None, top_out=.9, bottom_out=0.5):
        if overlap != .5 and overlap != 0.: raise NotImplementedError

        log.debug(f"Predicting w/ img of shape {img.shape}")

        slices = CustomImageDataGenerator.iter_slice(
                self.input_shape, img.shape,
                overlap=overlap, final_overlap=False)
        slices = list(slices)

        outshape = self.make_outputshape(img)
        yhat = np.zeros(outshape, dtype=img.dtype)
        window_shape = (img.shape[0], *self.input_shape, img.shape[-1])

        for xl, xu, yl, yu in slices:
            _yhat = self.predict_crop(img[:, xl:xu, yl:yu, :])
            inrange_assert(_yhat)
            _yhat *= (1. - overlap)  #TODO: edge effects
            yhat[:, xl:xu, yl:yu, :] += _yhat
            inrange_assert(yhat)

        if prediction_threshold is not None:
            yhat = (yhat > prediction_threshold).astype(data.DTYPE)

        if top_out:
            yhat[yhat > top_out] = 1.0

        if bottom_out:
            yhat[yhat < bottom_out] = 0.0

        if not top_out or not bottom_out:
            np.clip(yhat, 0, 1, out=yhat)

        return yhat

    def make_outputshape(self, img):
        return (*img.shape[:-1], self.output_channels)

