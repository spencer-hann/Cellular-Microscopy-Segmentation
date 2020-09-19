import signal
import logging
import time

from pathlib import Path
from collections import namedtuple
from contextlib import contextmanager
from functools import wraps
from queue import SimpleQueue
from itertools import cycle, islice

from matplotlib import pyplot as plt; plt.style.use("dark_background")

from . import Options


figures_folder = Path("/home/spencer/ohsu/research/em/figures/")


log = logging.getLogger(__name__)


@Options.using_options
def plot_save_name(model, round, epoch, opt=None):
    #return f"{type(model).__name__}_r{round}_e{epoch}"
    return f"{type(model).__name__}_{opt.save_model_filename}_e{epoch}"


def plotter_wrapper(func):
    @wraps(func)
    def wrapper(
        *args,
        show=False,
        block=False,
        tight_layout=True,
        allow_interupt=True,
        pause_time=.5,
        save='',
        prepend_figures_folder=True,
        **kwargs
    ):
        r = func(*args, **kwargs)

        if tight_layout:
            with silent_logger(20):
                plt.tight_layout()
        if show:
            try:
                with silent_logger(20):
                    plt.show(block=block)
                    plt.pause(pause_time)
            except KeyboardInterrupt:
                log.debug(
                    f"KeyboardInterrupt for plotter_wrapper:{func.__name__}"
                )
                if not allow_interupt: raise

        if save:
            if prepend_figures_folder:
                save = figures_folder / save
            elif not isinstance(save, Path):
                save = Path(save)
            msg = f"Saving figure {save.name}"
            log.info("*** " + msg + " ***")
            plt.savefig(fname=save)
            plt.close()

        return r

    return wrapper


class GeneratorSplitter:
    def __init__(self, generator, cyclic=False):
        if cyclic:
            self.source = cycle(generator)

        self.first = SimpleQueue()
        self.second = SimpleQueue()

        self.first_split_called = False
        self.second_split_called = False

    @classmethod
    def both(cls, generator, *args, **kwargs):
        splitter = cls(generator, *args, **kwargs)
        return splitter.first_split(), splitter.second_split()

    def first_split(self):
        if self.first_split_called:
            raise RuntimeError("Only call first_split once!")
        self.first_split_called = True

        while True:
            if self.first.empty():
                first, second = next(self.source)
                self.second.put(second); del second;
                first = HolderReleaser(first)
                yield first.release(); del first;
            else:
                yield self.first.get()

    def second_split(self):
        if self.second_split_called:
            raise RuntimeError("Only call first_split once!")
        self.second_split_called = True

        while True:
            if self.second.empty():
                first, second = next(self.source)
                self.first.put(first); del first;
                second = HolderReleaser(second)
                yield second; del second;
            else:
                yield self.second.get()


class GeneratorPeeker:
    def __init__(self, generator):
        self.source = generator
        self.hold = None

    def peek(self,):
        if self.hold is None:
            self.hold = HolderReleaser(next(self.source))
        return self.hold.item

    def __iter__(self):
        while True:
            if self.hold is not None:
                yield self.hold.release()
                self.hold = None
            else:
                yield next(self.source)


class HolderReleaser:
    """ A simple object capable of holding a single item and then returning that
    item such that no internal references are held.  Useful for holding large
    objects that should not linger in memory any longer than necessary. """
    def __init__(self, item=None):
        self.item = item

    def hold(self, item):
        self.item = item
        return item

    def release(self):
        temp = self.item
        self.item = None
        return temp


def timed_input(timeout=5, timeout_msg="Input timed out",):
    class _InputTimeout(BaseException): pass
    def timeout_occured(*args): raise _InputTimeout

    signal.signal(signal.SIGALRM, timeout_occured)
    time = signal.alarm(timeout)

    s = None
    try:
        s = input()
    except _InputTimeout:
        if timeout_msg: log.info(timeout_msg)

    signal.alarm(0)
    signal.signal(signal.SIGALRM, signal.SIG_IGN)

    return s


@contextmanager
def silent_logger(level=logging.ERROR, logger=None):
    if logger is None:
        logger = logging.getLogger()

    prev_level = logger.getEffectiveLevel()
    logger.setLevel(level)

    try:
        yield logger
    finally:
        logger.setLevel(prev_level)


def iter_chunks(iterable, n, wrap=lambda a: a):
    iterable = iter(iterable)
    chunk = wrap(islice(iterable, n))
    while chunk:
        yield chunk
        chunk = wrap(islice(iterable, n))


def make_stopwatch(start=None, as_string=True, round_seconds=None):
    if start is None:
        start = time.time()

    def stopwatch():
        t = time.time() - start
        if as_string:
            hours, rem = divmod(t, 3600)
            minutes, seconds = divmod(rem, 60)
            seconds = round(seconds, round_seconds)
            t = f"{int(hours)}:{int(minutes)}:{seconds}"
        return t

    return stopwatch

