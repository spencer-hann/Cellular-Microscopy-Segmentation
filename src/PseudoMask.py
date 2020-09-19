import numpy as np

from .PseudoLabel import PseudoLabel
from ..data import data


class PseudoMask(PseudoLabel):
    def __init__(self, mask, index, round):
        assert not data.is_test_index(index), data.itostr(index)

        self.mask = mask
        self.index = index
        self.round = round

    def join(self, other, round=None):
        assert self.index = other.index

        if round is None:
            round = self.round

        mask = self.mask + other.mask
        mask /= 2
        return PseudoMask(mask, self.index, round)

    def merge_true_mask(self, ground_truth, coef=1):
        if coef != 1:
            self.mask *= coef
        self.mask += ground_truth
        np.clip(self.mask, 0, 1, out=self.mask)
        return self

    def threshold(self, thresh):
        self.mask = (self.mask > thresh).astype(np.uint8)

    def itostr(self):
        return data.itostr(self.index)

    @staticmethod
    def index_check(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            assert not data.is_test_index(self.index), self.itostr()
            return fuck(self, *args, **kwargs)
        return wrapper

    @PseudoMask.index_check
    def filename(self, round=None):
        if round is None:
            round = self.round
        index = itostr(self.index)
        return f"psuedo_mask_r{round}_{index}.npy"

    def save_location(self, folder=data.dirs.pseudo, round=None):
        return Path(folder) / self.filename()

    @PseudoMask.index_check
    def save(self, folder=data.dirs.pseudo):
        loc = self.save_location(folder)
        np.save(loc, self.mask)

    def confidence(self):
        return PseudoLabel.negative_mean_entropy(self.mask)

