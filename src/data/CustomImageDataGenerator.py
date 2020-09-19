import tensorflow as tf
import numpy as np
import logging

import gc

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from IPython.display import clear_output
from itertools import islice, cycle
from random import randrange

from ..Utils import iter_chunks
from .Image import Image
from . import data

log = logging.getLogger(__name__)


class CustomImageDataGenerator(tf.keras.preprocessing.image.ImageDataGenerator):
    _counter = 0
    names = []
    def __init__(self, *args, name='', **kwargs):
        CustomImageDataGenerator._counter += 1
        name = name + str(CustomImageDataGenerator._counter)
        CustomImageDataGenerator.names.append(name)
        self.name = name
        super().__init__(*args, **kwargs)

    def __del__(self):
        #CustomImageDataGenerator._counter -= 1
        CustomImageDataGenerator.names.remove(self.name)

    def flow_from_indices(
        self,
        indices,
        shape=(512, 512),
        batch_size=8,
        overlap=.0,
        shuffle=True,
        make_null_mask=False,
        add_nuclei_to_cells=True,
        allow_test_data=False,
        preprocess=lambda x: x,  # identity
        total=float('inf'),
        **kwargs
    ):
        indices = data.index_checker(indices, "flow_from_indices", allow_test_data)
        indices = list(indices)
        outof = len(indices) // batch_size

        if len(indices) < batch_size:
            log.error(f"not enough items ({len(indices)}) to fill batch")

        if shuffle:
            np.random.shuffle(indices)

        loader = lambda i: (
            data.load_np_by_index(
                i,
                make_null_mask=make_null_mask,
                add_nuclei_to_cells=add_nuclei_to_cells,
                preprocess=preprocess,
            )
        )

        indices = iter_chunks(indices, batch_size, list)
        ibatch = next(indices)

        batch, count = None, 0
        while len(ibatch) == batch_size:
            clear_output(); count+=1;
            log.info(f"Beginning macro-batch {count}/{outof} ({self.name})")
            log.debug(f"Macro-batch: {ibatch}")

            del batch; gc.collect();  # release old memory first
            batch = np.stack([loader(i) for i in ibatch])
            batch = next(self.flow(batch, batch_size=batch.shape[0], **kwargs))

            crops = self.iter_crop(batch, shape, shuffle=True, overlap=overlap)
            for crop in crops:
                yield crop[..., :1], crop[..., 1:]  # image, masks
                #if num_yield >= total: return

            ibatch = next(indices)

    @staticmethod
    def iter_slice(shape, outershape, overlap=.0, final_overlap=True):
        x_crop, y_crop = shape
        x_shift = int(x_crop * (1 - overlap))  # overlap to crops
        y_shift = int(y_crop * (1 - overlap))

        if not final_overlap:
            outershape = list(outershape)
            outershape[1] -= (outershape[1] % x_shift)
            outershape[2] -= (outershape[2] % y_shift)

        x = outershape[1] - x_crop
        y_start = outershape[2] - y_crop
        del outershape

        while x > 0:
            y = y_start
            while y > 0:
                yield x, x+x_crop, y, y+y_crop
                y -= y_shift

            # TODO: delete asserts
            assert final_overlap or y == 0, f"y = {y}, {shape}, {outershape}"

            yield x, x+x_crop, 0, y_crop
            x-= x_shift

        assert final_overlap or x == 0, f"x = {x}, {shape}, {outershape}"

        y = y_start
        while y > 0:
            yield 0, x_crop, y, y+y_crop
            y -= y_shift

        assert final_overlap or y == 0, f"y = {y}, {shape}, {outershape}"

        yield 0, x_crop, 0, y_crop

    @staticmethod
    def iter_crop(a, shape, overlap=0., shuffle=True, final_overlap=True):
        slices = CustomImageDataGenerator.iter_slice(
                shape, a.shape, overlap, final_overlap)

        if shuffle:
            slices = list(slices)
            np.random.shuffle(slices)

        for xl, xu, yl, yu in slices:  # lower/upper indices
            yield a[:, xl:xu, yl:yu, :]

    @staticmethod
    def random_crop(a, crop_shape):
        xcrop = randrange(a.shape[1] - crop_shape[0])
        ycrop = randrange(a.shape[2] - crop_shape[1])
        return a[:, xcrop:xcrop+crop_shape[0], ycrop:ycrop+crop_shape[1], :]


    #def flow_from_generator(
    #    self, x, y=None, shape=(512, 512), append_void_mask=True, **kwargs
    #):
    #    if y is None:  # assume x is iterable of Image.MaskedImage objects
    #        #y = (image.mask for image in x)
    #        #x = (image.im for image in x)
    #        x, y = GeneratorSplitter.both(x, cyclic=True)

    #    batch_size = kwargs.pop("batch_size", 8)

    #    while True:
    #        try:
    #            batch = tf.concat((
    #                    tf.stack(list(islice(x, batch_size))),
    #                    tf.stack(list(islice(y, batch_size)))
    #                ), axis=3)
    #        except (RuntimeError, StopIteration):
    #            log.warning("end of data generator")
    #            return

    #        #batch = CustomImageDataGenerator.random_crop(batch, shape)
    #        batch = CustomImageDataGenerator.iter_crop(
    #            next(self.flow(batch, batch_size=batch.shape[0], **kwargs)),
    #            shape
    #        )
    #        assert batch.shape[0] == batch_size, batch.shape[0]

    #        for crop in batch:
    #            if append_void_mask:
    #                crop = self.append_void_mask(crop)
    #            yield (
    #                crop[:,:,:,:1] / 255.,
    #                crop[:,:,:,1:]
    #            )

    #@staticmethod
    #def append_void_mask(t):
    #    raise Exception("Don't use void mask w/ binary cross entropy loss")
    #    voidmask = np.empty((*t.shape[:-1], 1), dtype=bool)
    #    for i in range(t.shape[0]):
    #        voidmask[i,:,:,0] = tf.logical_not(
    #            # channel-wise logical or to get all masks
    #            tf.reduce_any(tf.cast(t[i,:,:,1:], bool), axis=-1)
    #        )
    #    return tf.concat((t, voidmask), axis=-1)

    #def to_tf_dataset(
    #    self,
    #    x,
    #    y=None,
    #    shape=(512,512),
    #    output_types=(tf.float32, tf.float32),
    #    output_shapes=None,
    #    autotune=True,
    #    repeat=False,
    #    **kwargs
    #):
    #    if output_shapes is None:
    #        output_shapes = ((None, *shape, 1), (None, *shape, 2))

    #    ds = tf.data.Dataset.from_generator(
    #        lambda: self.flow_from_generator(
    #            x, y=y, shape=shape, **kwargs),
    #        output_types=output_types,
    #        output_shapes=output_shapes,
    #    )

    #    if repeat:
    #        log.debug("setting dataset to repeat")
    #        ds = ds.repeat()
    #    if autotune:
    #        log.debug("setting dataset to autotune prefetching")
    #        ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    #    return ds

