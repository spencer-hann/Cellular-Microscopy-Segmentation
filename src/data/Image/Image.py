import numpy as np
import tensorflow as tf

from ...Utils import plotter_wrapper, silent_logger

from matplotlib import pyplot as plt
from PIL import Image as PIL_Image
from pathlib import Path


class Image:
    def __init__(self, path, **kwargs):
        if isinstance(path, str): path = Path(str)
        self.im = Image.tensor_from_path(path, **kwargs)
        self.index = Image.parse_index(path)
        self.path = path

        # TODO: delete after checking to see if this class is in use
        raise Exception(str((self.index, self.path)))

    @property
    def shape(self):
        return self.im.shape

    @property
    def dtype(self):
        return self.im.dtype

    def copy_image(self, image):
        self.im = image.im
        self.index = image.index
        return self

    @staticmethod
    def tensor_from_path(path,):
        with silent_logger(30):  # 30 = logging.WARNING
            arr = np.array(PIL_Image.open(path))[:,:,None]
        return tf.convert_to_tensor(arr)

    @staticmethod
    def parse_index(path): # expects file name w/ 4-digit id/index before suffix
        return np.uint16(path.stem[-4:])

    def add_mask(self, mask, names=None):
        return MaskedImage(self, mask, names)


class MaskedImage(Image):
    def __init__(self, img, mask, mask_names=None, check_index=True, **kwargs):
        self.crop_shape = np.array((256,256))
        if isinstance(img, Image):
            self.copy_image(img)
        elif isinstance(img, Path):
            super().__init__(img)
        else:
            raise RuntimeError(f"img arg has unsupported type {type(img)}")

        if isinstance(mask, Path):
            self.mask = Image.tensor_from_path(make, **kwargs)
            index = Image.parse_index(mask)
            if check_index: assert self.index == index, (self.index, index)
        elif isinstance(mask, Image):
            self.mask = mask.im
            if check_index: assert self.index == mask.index, (self.index, mask.index)
        elif isinstance(mask, tf.Tensor):
            self.mask = mask
        else:
            raise RuntimeError(f"mask arg has unsupported type {type(img)}")

        if isinstance(mask_names, str) or mask_names is None:
            mask_names = [mask_names,]
        self.names = mask_names

    def add_mask(self, mask, name=None):
        self.names.append(name)
        self.mask = tf.concat((self.mask, mask), axis=-1)
        return self

    def colored_image(self):
        cshape = list(self.shape)
        cshape[-1] = 3  # color dimension from grey-scale to RGB
        img = np.ndarray(cshape, dtype = np.float32)
        img[...] = self.im.numpy()#[:,:,None]
        img /= img.max()
        mask = tf.squeeze(self.mask).numpy().astype(bool)
        if mask.ndim == 2:
            mask = mask[:,:,np.newaxis]

        for i in range(mask.shape[2]):
            m = mask[:,:,i]
            print(mask.shape, img.shape)
            print(m.shape, img[:,:,i].shape)
            img[:,:,i][m] *= 1.4

        return img

    def tensor(self):
        return tf.stack(self.im, self.mask, axis=0)

    def __iter__(self):
        yield self.im; yield self.mask;

    def add_mask(self, mask, names=None):
        if isinstance(names, str):
            names = (names,)

        if type(mask) is Image:
            mask = mask.im
        elif type(mask) is MaskedImage:
            mask = mask.mask

        self.names = (*self.names, *names)
        self.mask = tf.stack((self.mask, mask), 2)

        return self

    @plotter_wrapper
    def highlight_plot(self):
        img = self.colored_image()
        plt.imshow(img)
        plt.title(str(self.index).zfill(4))
        return plt.gca()

