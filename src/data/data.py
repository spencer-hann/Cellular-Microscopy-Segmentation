import numpy as np

from .Image import Image

from pathlib import Path
from collections import namedtuple


_data_dir = Path.cwd() / "data"

dirs = namedtuple(
    "DataDirTuple",
    ("main", "images", "mitochondria", "nuclei",)
)(
    _data_dir,
    _data_dir / "Images",
    _data_dir / "Mitochondria",
    _data_dir / "Nuclei",
)


def is_valid_imagefile(f):
    return f.suffix != ".info" and not f.name.startswith('.')


def load_images(path=dirs.images, n=-1,):
    files = (f for f in path.iterdir() if is_valid_imagefile(f))

    if n > -1:
        files = (next(files) for _ in range(n))

    return (Image(f) for f in files)


def itostr(index):
    return str(index).zfill(4)


def _load_by_index(folder, fileprefix, index):
    path = folder / f"{fileprefix}{itostr(index)}.tif"
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

