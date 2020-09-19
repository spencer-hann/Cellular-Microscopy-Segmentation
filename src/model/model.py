import tensorflow as tf

from tensorflow_examples.models.pix2pix import pix2pix

from itertools import islice
from matplotlib import pyplot as plt

from ..Utils import Options


class MobileUNet(tf.keras.Model):
    @Options.using_options
    def __init__(
        self,
        opt=None,
        input_channels=1,
    ):
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=(*opt.input_shape, 3), include_top=False, weights=opt.weights)

        # Use the activations of these layers
        layer_names = (
            'block_1_expand_relu',   # 64x64
            'block_3_expand_relu',   # 32x32
            'block_6_expand_relu',   # 16x16
            'block_13_expand_relu',  # 8x8
            'block_16_project',      # 4x4
        )
        layers = [base_model.get_layer(name).output for name in layer_names]

        # Create the feature extraction model
        down_stack = tf.keras.Model(inputs=base_model.input, outputs=layers)

        up_stack = (
            pix2pix.upsample(512, 4, apply_dropout=True),  # 4x4 -> 8x8
            pix2pix.upsample(512, 4, apply_dropout=True),  # 8x8 -> 16x16
            pix2pix.upsample(256, 4, apply_dropout=True),  # 16x16 -> 32x32
            pix2pix.upsample(64, 4),   # 32x32 -> 64x64
        )

        inputs = tf.keras.layers.Input(shape=(*opt.input_shape, input_channels))
        # adapt input_channels to 3 for MobileNet
        x = tf.keras.layers.Conv2D(3, 3, padding='same')(inputs)

        # Downsampling through the model
        skips = down_stack(x)
        x = skips[-1]
        skips = reversed(skips[:-1])

        # Upsampling and establishing the skip connections
        for up, skip in zip(up_stack, skips):
            x = up(x)
            concat = tf.keras.layers.Concatenate()
            x = concat([x, skip])

        # This is the last layer of the model
        last = tf.keras.layers.Conv2DTranspose(
            opt.output_channels, 4, strides=2,
            activation=opt.final_activation,
            padding='same')  #64x64 -> 128x128

        x = last(x)

        super().__init__(inputs=inputs, outputs=x, name=type(self).__name__)

    def plot_model(self, to_file=None, show_shapes=True, **kwargs):
        if to_file is None:
            to_file = type(self).__name__ + ".png"
        return tf.keras.utils.plot_model(self, to_file, show_shapes, **kwargs)

