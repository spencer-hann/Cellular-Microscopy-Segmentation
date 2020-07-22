import numpy as np
import tensorflow as tf

from matplotlib import pyplot as plt
from PIL import Image as PIL_Image
from pathlib import Path


class Image:
    def __init__(self, path):
        if isinstance(path, str): path = Path(str)
        self.im = Image.tensor_from_path(path)
        self.index = Image.parse_index(path)
        self.path = path

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
    def tensor_from_path(path):
        return tf.convert_to_tensor(np.array(PIL_Image.open(path)))

    @staticmethod
    def parse_index(path): # expects file name w/ 4-digit id/index before suffix
        return np.uint16(path.stem[-4:])

    def add_mask(self, mask, names=None):
        return MaskedImage(self, mask, names)


class MaskedImage(Image):
    def __init__(self, img, mask, mask_names=None, check_index=True):
        if isinstance(img, Image):
            self.copy_image(img)
        elif isinstance(img, Path):
            super().__init__(img)
        else:
            raise RuntimeError(f"img arg has unsupported type {type(img)}")

        if isinstance(mask, Path):
            self.mask = Image.tensor_from_path(make)
            index = Image.parse_index(mask)
            if check_index: assert self.index == index, (self.index, index)
        elif isinstance(mask, Image):
            self.mask = mask.im
            if check_index: assert self.index == mask.index, (self.index, mask.index)
        elif isinstance(mask, tf.Tensor):
            self.mask = mask
        else:
            raise RuntimeError(f"mask arg has unsupported type {type(img)}")

        if isinstance(mask_names, str):
            mask_names = (mask_names,)
        self.names = mask_names

    def colored_image(self):
        cshape = (*self.shape, 3)
        img = np.ndarray(cshape, dtype = np.float32)
        img[...] = self.im.numpy()[:,:,None]
        img /= img.max()
        mask = self.mask.numpy().astype(bool)
        if mask.ndim == 2:
            mask = mask[:,:,np.newaxis]
        print(mask.shape)

        for i in range(mask.shape[2]):
            m = mask[:,:,i]
            img[:,:,i][m] *= 1.4

        return img

    def add_mask(self, mask, names=None):
        if isinstance(names, str):
            names = (names,)

        if type(mask) is Image:
            print(mask.path)
            mask = mask.im
        elif type(mask) is MaskedImage:
            mask = mask.mask

        self.names = (*self.names, *names)
        self.mask = tf.stack((self.mask, mask), 2)

        return self

    def highlight_plot(self, show=True):
        img = self.colored_image()
        plt.imshow(img)
        plt.title(str(self.index).zfill(4))

        if show:
            plt.show()

