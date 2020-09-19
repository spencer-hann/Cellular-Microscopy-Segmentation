import tensorflow as tf

from tensorflow.keras import losses


from ..data.CustomImageDataGenerator import CustomImageDataGenerator


def entropy_weights(n):
    for weight in range(n):
        weight /= n
        yield weight**2
    while True:
        yield 1.


class ImageEntropy(losses.Loss):
    def __init__(self, *args, loss_weights=(1.,1.), **kwargs):
        super().__init__(*args, **kwargs)

        self.loss_weights = tf.convert_to_tensor(loss_weights, dtype=tf.float32)
        self.loss_weights /= float(sum(loss_weights))

    @staticmethod
    def _tf_mean_entropy(T, base='e', axes=(1,2)):
        # minimizing entropy should drive ouputs closer to 1 or 0
        T = tf.clip_by_value(T, 1e-4, 1.)
        log = tf.math.log(T)
        if base != 'e':
            log /= tf.math.log(float(base))
        ent = tf.math.multiply_no_nan(T, log)
        ent = tf.reduce_mean(ent, axis=axes)
        ent = tf.negative(ent)
        return ent

    def call(self, y_true, y_pred):
        if 0:  # TODO experiment:
            # bump y_true false labels to help w/ false false positives
            # (false negatives in data due to incomplete masks)
            t_true = tf.add(y_true, .2, name="false neg bump?")
            t_true = tf.clip_by_value(y_true, 0, 1, name="clip after bump")

        loss = losses.binary_crossentropy(y_true, y_pred)
        # TODO: should entropy be w/ log base 2?
        entropy = ImageEntropy._tf_mean_entropy(y_pred, axes=(1,2,3))

        std = ImageEntropy.sliding_stddev(y_pred)
        pos = ImageEntropy.positive_cross_entropy(y_true, y_pred)

        loss = tf.reduce_mean(loss, axis=(1,2))

        #loss = tf.multiply(loss, self.loss_weights[0], name="weight_bin_crossentropy")
        #entropy = tf.multiply(entropy, self.loss_weights[1], name="weight_entropy")
        #pos = tf.multiply(entropy, self.loss_weights[2], name="weight_entropy")

        return tf.math.accumulate_n((
            loss,
            entropy,
            pos,
            std,
        ),)

    def sliding_stddev(T, overlap=.5):
        shape = (64, 64)

        #total = tf.zeros(T.shape[0], dtype=tf.float32)
        total = []
        crops = CustomImageDataGenerator.iter_crop(
                T, shape, overlap, shuffle=False)
        for i, crop in enumerate(crops):
        # check tf.image.crop_to_bounding_box
                channelwise = tf.math.reduce_std(crop, axis=(1,2))
                mean_std = tf.reduce_mean(channelwise, axis=1)
                total.append(mean_std)

        #total = tf.math.accumulate_n(total)
        #total = tf.divide(total, float(i))
        total = tf.reduce_mean(total, axis=0)  # shape is (n_crops, batch_size)
        return total

    def positive_cross_entropy(y_true, y_pred):
    # just half of binary crossentropy
    # optimizes for recall/sensitivity
        y_pred = tf.clip_by_value(y_pred, 1e-4, 1.)
        log = tf.math.log(y_pred)
        r = tf.math.multiply_no_nan(y_true, log)
        r = tf.reduce_mean(r, axis=(1,2,3))
        return tf.negative(r)

