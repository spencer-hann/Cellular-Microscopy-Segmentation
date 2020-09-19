import tensorflow as tf
import logging

from argparse import ArgumentParser
from pathlib import Path
from collections import namedtuple, OrderedDict
from time import strftime
from functools import wraps


log = logging.getLogger(__name__)


global_opt = None


defs = OrderedDict(
    log="INFO",
    logfmt='%(levelname)s::%(module)s.%(funcName)s:%(lineno)d: %(message)s',
    input_shape=(512, 512),
    output_channels=3,
    val_split=0.15,
    epochs=40,
    rounds=3,
    steps_per_epoch=256,
    val_steps_per_epoch=128,
    lr=0.004,
    overlap=.2,
    batch_size=12,  # 16
    val_batch_size=2,
    final_activation="sigmoid",
    weights='None',
    prefetch_buffer_size='1',
    save_model_filename='None',
    load_model_filename='None',
    parent_model_filename=('None',),
    models_folder='/home/spencer/ohsu/research/em/models/',
    confidence_threshold=.9,

)

parser = ArgumentParser()
for option, value in defs.items():
    t = type(value)
    n = None
    if t is tuple:
        t = type(value[0])
        n = '+'
    parser.add_argument("--" + option, default=value, type=t, nargs=n)


defs = namedtuple("OptDefaultsTuple", defs.keys())(*defs.values())


def get_options(
    auto_set=True,
    include_defaults=False,
    overrides={},
    show=True,
    show_level="INFO",
    show_time=True,
    autotune_warning=True,
):
    global global_opt
    if global_opt is not None:
        return global_opt

    opt = parser.parse_args()

    for name, value in overrides.items():
        setattr(opt, name, value)

    opt.log = opt.log.upper()
    if not opt.log.isnumeric():
        opt.log = getattr(logging, opt.log)

    if len(opt.input_shape) == 1:
        opt.input_shape = (opt.input_shape, opt.input_shape)

    none_check = lambda a: None if a == "None" else a
    opt.weights = none_check(opt.weights)
    opt.save_model_filename = none_check(opt.save_model_filename)
    opt.load_model_filename = none_check(opt.load_model_filename)
    opt.parent_model_filename = list(map(none_check, opt.parent_model_filename))

    opt.models_folder = Path(opt.models_folder)

    if opt.prefetch_buffer_size.upper() == "AUTOTUNE":
        opt.prefetch_buffer_size = tf.keras.experimental.AUTOTUNE
        if autotune_warning:
            log.warning("Using experimental AUTOTUNE buffer size for prefetch.")
    else:
        opt.prefetch_buffer_size = int(opt.prefetch_buffer_size)

    if auto_set:
        logging.basicConfig(level=opt.log, format=opt.logfmt)

    if show:
        if show_time:
            getattr(log, show_level.lower())(strftime("%X %x %Z"))
        getattr(log, show_level.lower())(opt)

    if include_defaults:
        opt.defaults = defs

    global_opt = opt
    return opt


def using_options(func):
    @wraps(func)
    def _using_options(*args, **kwargs):
        kwargs['opt'] = kwargs.pop('opt', get_options())
        return func(*args, **kwargs)

    return _using_options


#parser.add_argument("--log", default=defs.log)
#parser.add_argument("--logfmt", default=defs.logfmt)
#parser.add_argument("--input_shape", default=defs.input_shape, type=int, nargs="+",)
#parser.add_argument("--output_channels", default=defs.output_channels, type=int)
#parser.add_argument("--val_split", default=defs.val_split, type=float)
#parser.add_argument("--epochs", default=defs.epochs, type=int)
#parser.add_argument("--steps_per_epoch", default=defs.steps_per_epoch, type=int)
#parser.add_argument("--val_steps_per_epoch", default=defs.val_steps_per_epoch, type=int)
#parser.add_argument("--lr", default=defs.lr, type=float)
#parser.add_argument("--overlap", default=defs.overlap, type=float)
#parser.add_argument("--batch_size", default=defs.batch_size, type=int)
#parser.add_argument("--final_activation", default=defs.final_activation)
#parser.add_argument("--weights", default=defs.weights)

