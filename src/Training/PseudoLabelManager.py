import tensorflow as tf
import numpy as np

import logging

from scipy.stats import entropy

from .PseudoLabel import PseudoLabel
from ..data import data
from ..Utils import iter_chunks


log = logging.getLogger(__name__)


class PseudoLabelManager:
    def __init__(
        self, data_func, *args, batch_size=None, **kwargs
    ):
        self.starter = (data_func, args, kwargs)
        self.current = self.starter
        self._repeat = False
        self.batch_size = batch_size
        self._current_flow = None

    def __iter__(self):
        self.__iter__ = self.flow
        return self.flow()

    def _create_stop_condition(self):
        def _stop_condition():
            if not _stop_condition.called:
                _stop_condition.called = True
                return False  # go once without stopping
            return not self._repeat  # defer to self._repeat

        _stop_condition.called = False
        return _stop_condition

    # needed for memory leaks because tensorflow
    def _kill_current_flow(self):
        if self._current_flow is None:
            return
        self.discontinue = True
        temp = None
        try:
            temp = next(self._current_flow)
        except StopIteration:
            pass
        self._current_flow = None
        return temp

    def flow(self, stop_condition=None, batch_size=None):
        self._kill_current_flow()  # there can be only one
        self.discontinue = False
        self._current_flow = self._flow(stop_condition, batch_size)
        return self._current_flow

    def _flow(self, stop_condition, batch_size):
        if stop_condition is None:  # use default
            stop_condition = self._create_stop_condition()
        elif isinstance(stop_condition, int):  # bool
            b = stop_condition
            stop_condition = lambda: b

        if batch_size is None:
            batch_size = self.batch_size

        f, args, kwargs = self.current

        while not stop_condition():
            data_iter = f(*args, **kwargs)

            if batch_size is not None:
                mkarr = lambda islice: np.array((*islice,))
                data_iter = iter_chunks(data_iter, batch_size, mkarr)

            for datum in data_iter:
                yield datum
                if self.discontinue:
                    log.warning(f"flow in {type(self).__name__} killed.")
                    self._current_flow = None
                    return

    def refresh(self, func, *args, **kwargs):
        if func is None:
            func = self.generate_pseudo_data
        self.current = func, args, kwargs
        return self

    def generate_pseudo_data(self, *args, **kwargs):
        f, fargs, fkwargs = self.starter
        data_iter = f(*fargs, **fkwargs)
        return self._generate_pseudo_data(data_iter, *args, **kwargs)

    @staticmethod
    def confidence_matrix(x, base=2, epsilon=1e-5):
        axes = (1, 2) # ignore N and channel dims, take HxW entropy
        log = np.log(np.clip(x, epsilon, 1))  # use np.log2?
        if base != 'e': log /= np.log(base)
        ent = x * log  # element-wise entropy
        ent = ent.mean(axis=axes)  # mean entropy per image/channel
        return ent + 1.

    @staticmethod
    def _generate_pseudo_data(
        data_iter,
        model,
        threshold,
        confidence_func=None,
        merge_ground_truth=True,
        log_confidence=True,
        clip_vals=(0,1),
    ):
        log.debug("Generating pseudo data")
        if confidence_func is None:
            confidence_func = PseudoLabelManager.confidence_matrix

        n = 256
        conf_tracker, count = 0., 0
        accepted = 0.

        for x, y in data_iter:
            yhat = model.predict(x)

            np.sqrt(yhat, out=yhat)  # lazy gamma correction

            # plain confidences by image/channel
            conf_matrix = confidence_func(yhat)
            conf_tracker += conf_matrix.mean()
            count += 1
            if count >= 256:
                conf_tracker /= n * y.shape[0] * y.shape[-1]
                log.debug(f"Mean confidence {conf_tracker}")
                conf_tracker = 0.
                log.debug(f"# Pseudo masks accepted {accepted / count} {accepted}")
                accepted = 0.
                count = 0

            # confidence threshold mask
            conf_matrix = (conf_matrix > threshold).astype(data.DTYPE)
            accepted += conf_matrix.mean()  # mean takes for batch/channels

            # broadcasting NC array w/ NHWC array...
            yhat *= conf_matrix[:,None,None,:]; del conf_matrix

            np.around(yhat, out=yhat)

            yhat += y.numpy() if tf.is_tensor(y) else y; del y

            np.clip(yhat, *clip_vals, out=yhat)

            yield x, yhat
            del x, yhat


    @staticmethod
    def _old_generate_pseudo_data(
        data_iter,
        model,
        threshold,
        confidence_func=PseudoLabel.negative_mean_entropy,
        merge_ground_truth=True,
        log_confidence=True,
        clip_vals=(0,1),
    ):
        conf_tracker = np.array([0., 0.])

        for xbatch, ybatch in data_iter:
            yhatbatch = model.predict(xbatch)

            yout = np.empty(ybatch.shape, dtype=data.DTYPE)

            for i, (y, yhat) in enumerate(zip(ybatch, yhatbatch)):
                confidence = confidence_func(yhat)
                conf_tracker += [confidence, 1.]
                if confidence > threshold:
                    if merge_ground_truth:
                        y += yhat
                        del yhat
                        out = None
                        if tf.is_tensor(y): out = y.numpy()
                        elif isinstance(y, np.ndarray): out = y
                        y = np.clip(y, *clip_vals, out=out)
                        #if tf.is_tensor(y): y = y.numpy()
                        #np.clip(y, *clip_vals, out=y)
                    else:
                        y = yhat
                        del yhat

                yout[i,...] = y[...]
                del y

            yield xbatch, yout
            del xbatch, yout

        if log_confidence:
            mean = conf_tracker[0] / conf_tracker[1]
            log.info(f"Mean confidence {mean}")

    @staticmethod
    def stop_after(n):
        counter = iter(range(n+1))
        return lambda: next(counter) != n

    def repeat(self, repeat=True):
        self._repeat = repeat
        return self

