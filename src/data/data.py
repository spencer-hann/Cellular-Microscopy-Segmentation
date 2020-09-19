import numpy as np
import tensorflow as tf

import logging
import random

from .Image import Image, MaskedImage
from .CustomImageDataGenerator import CustomImageDataGenerator
from ..Utils import plotter_wrapper, HolderReleaser, silent_logger, Options

from PIL import Image as PIL_Image
from pathlib import Path
from itertools import islice
from functools import wraps
from collections import namedtuple
from matplotlib import pyplot as plt


log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


raw_img_shape = (3511, 5728)
cell_mask_range = (204, 2243 + 204)  # for images w/ cell masks
test_range = (204, 600)
n_raw_images = 2526

DTYPE = np.float32


_data_dir = Path.cwd() / "data"
checkpoints_dir = Path.cwd() / "model_checkpoints"

mask_names = ("cells", "mitochondria", "nuclei", "nucleoli")

dirs = namedtuple(
    "DataDirTuple",
    ("main", "psuedo", "images", *mask_names)
)(
    _data_dir,
    _data_dir / "psuedo",
    _data_dir / "Images",
    _data_dir / "together1",
    _data_dir / "Mitochondria",
    _data_dir / "Nuclei",
    _data_dir / "Nucleoli",
)


mask_dirs = (dirs.cells, dirs.mitochondria, dirs.nuclei, dirs.nucleoli)


file_prefixes = namedtuple(
    "FilePrefixTuple",
    ("images", *mask_names)
)(
    "SMMART_101b.raw_edge_crop.contrast_norm",
    "final_masks_",
    "SMMART_101b_Labels_mitochondria_Merged_",
    "101bNuclei_labels_",
    "Labels_SMMART_101b.Nucleoli_"
)


_ImageDataGenerator_defaults = dict(
    horizontal_flip=True,
    vertical_flip=True,
    shear_range=2,
    rotation_range=90,
    fill_mode='reflect',
    zoom_range=.1,
    width_shift_range=.2,
    height_shift_range=.2,
)


def is_test_index(index):
    return test_range[0] <= index < test_range[1]
# `test_range` cannot change
assert len(test_range) == 2
assert test_range[0] == 204
assert test_range[1] == 600


def expell_test_indices(indices):
    return (i for i in indices if not is_test_index(i))


def is_valid_imagefile(f):
    return f.suffix != ".info" and not f.name.startswith('.')


def all_training_indices():
    return expell_test_indices(range(n_raw_images))


def index_checker(indices, callers_name='', allow_test_data=False, warn=True):
    if not allow_test_data:
        indices = expell_test_indices(indices)
    elif warn:
        log.warning(
            "Allowing testing data"
            + (f" in {callers_name}" if callers_name else '')
        )
    return indices


def index_check(func):
    @wraps(func)
    def wrapper(index, *args, allow_test_data=False, **kwargs):
        if allow_test_data is False:
            assert not data.is_test_index(index), index
        return func(index, *args, allow_test_data=allow_test_data, **kwargs)
    return wrapper


def load_images(path=dirs.images, n=-1, shuffle=True, sort=False, **kwargs):
    files = _file_iter(path, n, shuffle, sort)
    return (Image(f, **kwargs) for f in files)


#def load_images_w_masks(path, n=-1, shuffle=True, **kwargs):
#    files = _file_iter(path, n, shuffle)
#
#    masks = (Image(f, **kwargs) for f in files)
#    return (
#        MaskedImage(
#            dirs.images / f"{file_prefixes.images}{itostr(mask.index)}.tif",
#            mask,)
#        for mask in masks
#    )


def _file_iter(directory, n, shuffle, sort=False):
    files = (f for f in directory.iterdir() if is_valid_imagefile(f))
    if n > -1: files = islice(files, n)
    if sort:
        files = sorted(list(files))
    if shuffle:
        files = list(files)
        random.shuffle(files)
    return files


def load_images_w_masks(
    names=mask_names,
    exclude=(),
    n=-1, shuffle=True, sort=False, **kwargs
):
    images = load_images(dirs.images, n, shuffle, sort, **kwargs)
    names = [name for name in names if name not in exclude]
    prefixes = [
        (getattr(dirs, mask_type), getattr(file_prefixes, mask_type))
        for mask_type in names
    ]

    for image in images:
        index = image.index
        paths = [folder / make_fname(prfx, index) for folder, prfx in prefixes]
        if not all(path.exists() for path in paths): continue
        masks = tf.concat([Image.tensor_from_path(p) for p in paths], axis=-1)
        image = image.add_mask(masks, names)
        yield image


def default_ImageDataGenerator(**kwargs):
    return CustomImageDataGenerator(  # kwargs override defaults
        {**_ImageDataGenerator_defaults, **kwargs})


def itostr(index):
    return str(index).zfill(4)


class TestDataException(BaseException):
    pass


def make_fname(fileprefix, index, cells=False, allow_test_data=False):
    if not allow_test_data and is_test_index(index):
        raise TestDataException(itostr(index) + " is from testing data!")

    if cells:  # index offset for files in 'together1' folder
        index = int(index) - 204

    index = itostr(index)
    return f"{fileprefix}{index}.tif"


def _load_by_index(folder, fileprefix, index):
    path = folder / make_fname(fileprefix, index)
    assert path.exists(), path
    return Image(path)


def load_image_by_index(index):
    return _load_by_index(dirs.images,
            "SMMART_101b.raw_edge_crop.contrast_norm", index)


def load_nuclei_by_index(index):
    return _load_by_index(dirs.nuclei, "101bNuclei_labels_", index)


def load_mitochondria_by_index(index):
    return _load_by_index(dirs.mitochondria,
            "SMMART_101b_Labels_mitochondria_Merged_", index)


@plotter_wrapper
def data_check_plot(image_gen=None, figsize=(24,16), shape=(4,4), **kwargs):
    fig = plt.figure(figsize=figsize)
    n = shape[0] * shape[1]
    images = list(islice(load_images(step=4), n))
    indices = [image.index for image in images]
    images = np.array([image.im.numpy() for image in images])
    log.debug(f"shape {images.shape}")

    if image_gen is None:
        image_gen = tf.keras.preprocessing.image.ImageDataGenerator(
            **_ImageDataGenerator_defaults
        )

    image_gen.fit(images)
    batch = next(image_gen.flow(images, batch_size=n, shuffle=False))
    for i, img in enumerate(batch):
        assert img.shape[2] == 1, img.shape
        img = img[:,:,0]
        if not isinstance(img, np.ndarray):
            img = img.numpy()
        ax = fig.add_subplot(*shape, i+1)
        ax.title.set_text(str(indices[i]))
        log.debug(i, indices[i])
        ax.imshow(img, **kwargs, )

    return fig


def check_highlight_plot():
    images = load_images()
    image = next(images)
    mask = load_nuclei_by_index(image.index)
    image = image.add_mask(mask, "nuclei")
    mask = load_mitochondria_by_index(image.index)
    image = image.add_mask(mask, "mitochondria")

    image.highlight_plot(show=True)


def check_image_data_generator(n=10):
    image_gen = default_ImageDataGenerator()

    images = load_images_w_masks(dirs.mitochondria)
    #images = islice(images, n)

    dataset = image_gen.to_tf_dataset(
            images, shape=(512,512), shuffle=False, repeat=False)

    images = iter(dataset)
    test_img, _ = next(images)

    log.debug(type(test_img))
    log.debug(test_img.shape)

    for i, (img, _) in enumerate(images):
        equal = (test_img == img).numpy().all()
        log.debug(i, ':', img.shape, 'are equal:', equal)
        assert not equal


def check_index(index):
    path = dirs.images / make_fname(file_prefixes.images, index)
    if path.exists():
        return path
    return None


def load_np_by_index(
    index, make_null_mask=False, add_nuclei_to_cells=True, preprocess=lambda a: a,
):
    nuclei = np_from_path(dirs.nuclei / make_fname(file_prefixes.nuclei, index))
    return np.concatenate(
        list(map(preprocess, (
            np_from_path(dirs.images / make_fname(file_prefixes.images, index)) / 255.,
            make_cell_mask(
                dirs.cells / make_fname(file_prefixes.cells, index, cells=True,),
                nuclei if add_nuclei_to_cells else None,
                make_null_mask=make_null_mask,
            ),
            #np_from_path(dirs.mitochondria / make_fname(file_prefixes.mitochondria, index)),
            nuclei,
            np_from_path(dirs.nucleoli / make_fname(file_prefixes.nucleoli, index)),
        ))),
        axis=-1,
    )


def make_cell_mask(
    path, nuclei_mask=None, shape=(*raw_img_shape, 1), normalize=True, make_null_mask=False
):
    if isinstance(path, np.ndarray):  # to shortcut loading from file
        cell_mask = path
    elif tf.is_tensor(path):
        cell_mask = path.numpy()
    elif Path(path).exists():  # otherwise must be actual Path/str
        cell_mask = np_from_path(path)
    elif make_null_mask:
        cell_mask = np.zeros(shape, dtype=DTYPE)
    else:
        raise RuntimeError(f"DNE: {str(path)}")

    if normalize:
        cell_mask /= 255.

    if nuclei_mask is not None:
        cell_mask += nuclei_mask
        np.clip(cell_mask, 0, 1, out=cell_mask)

    return cell_mask


def np_from_path(path, dtype=DTYPE, **kwargs):
    with silent_logger(logging.INFO):
        return np.array(PIL_Image.open(path), dtype=dtype, **kwargs)[:,:,None]


@Options.using_options
def index_split(indices, val_split=None, opt=None):
    if val_split is None:
        val_split = opt.val_split
    cutoff = int(len(indices) * val_split)
    indices = indices.copy()
    np.random.shuffle(indices)
    return indices[cutoff:], indices[:cutoff]


@Options.using_options
def make_training_set(
    indices=None, opt=None, output_shapes=None, name='train', **kwargs
):
    image_gen = CustomImageDataGenerator(
        name=name,
        horizontal_flip=True,
        vertical_flip=True,
        #shear_range=2,
        #rotation_range=20,#180,
        #fill_mode='reflect',
        #zoom_range=.1,
        #width_shift_range=.1,
        #height_shift_range=.1,
    )

    if indices is None:
        indices = np.arange(*cell_mask_range)

    if output_shapes is None:
        output_shapes = (
            (None, *opt.input_shape, 1),
            (None, *opt.input_shape, opt.output_channels)
        )

    train = image_gen.flow_from_indices(
        indices,
        opt.input_shape,
        opt.batch_size,
        opt.overlap,
        True,
        True,
        **kwargs
    )

    return train


@Options.using_options
def make_validation_set(
    indices, opt=None, name="val", **kwargs
):
    validation_gen = CustomImageDataGenerator(
        horizontal_flip=True, vertical_flip=True, name=name
    )
    return validation_gen.flow_from_indices(
        # TODO: do not allow null masks in validation data
        indices,
        opt.input_shape,
        batch_size=opt.val_batch_size,
        overlap=0.,
        make_null_mask=True,
        **kwargs
    )


@Options.using_options
def make_datasets(opt=None):
    indices = np.arange(*cell_mask_range)
    itrain, ival = index_split(indices)
    return make_training_set(itrain), make_validation_set(ival)


class TrainValSplitManager:
    def __init__(self, indices):
        self.indices = np.array(indices)
        self.refresh()

    def refresh(self):
        self.train, self.val = index_split(self.indices)
        return self

    def make_slices(self, sliceable):
        return sliceable[self.train], sliceable[self.val]

    @property
    def splits(self):
        return self.train, self.val


def make_master_dir(mask_names=mask_names, n=-1):
    imfiles = _file_iter(dirs.images, shuffle=False, sort=True, n=n)
    prefixes = [
        (getattr(dirs, name), getattr(file_prefixes, name))
        for name in mask_names
    ]

    def load(f):
        with silent_logger(logging.WARNING):
            return np.array(PIL_Image.open(f))[:,:,None]

    format_metadata = lambda row: ", ".join(row)
    metadata = [format_metadata(["index", *mask_names])]

    total = 0
    missing_masks = 0
    invalid_images = 0

    for f in imfiles:
        index = itostr(Image.parse_index(f))
        save_loc = dirs.master / f"master{index}.npy"
        if save_loc.exists():
            continue

        log.debug(f"Loading {str(f)}")
        image = load(f)
        paths = [folder / make_fname(prfx, index) for folder, prfx in prefixes]

        metadata.append([index,])  # define row
        masks = []

        mask = np.zeros_like(image)

        inval_img_flag = False
        for i, path in enumerate(paths):
            if path.exists():
                metadata[-1].append('1')  # valid mask
                #masks.append(load(path))
                m = load(path)
                mask[m] = i+1
            else:
                metadata[-1].append('0')  # emtpy/invalid mask
                #masks.append(np.zeros_like(image))
                log.warning(f"***Not Found***: {str(path)}")
                missing_masks += 1
                if not inval_img_flag:
                    inval_img_flag = True
                    invalid_images += 1

        metadata[-1] = format_metadata(metadata[-1])

        image = np.concatenate((image, mask), axis=-1)
        #image = np.concatenate((image, *masks,), axis=-1)

        log.info(f"Saving ({index}) {save_loc}")
        np.save(save_loc, image)
        total += 1

    with open(dirs.master / "info.txt", 'w') as f:
        f.write('\n'.join(metadata))

    log.info(
        f"{total} images total. w/ {invalid_images} "
        + "images missing at least one mask. "
        + f"{missing_masks} total missing masks. "
    )


