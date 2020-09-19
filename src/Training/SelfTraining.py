import logging

from .PseudoLabel import PseudoLabel
from .PseudoLabelManager import PseudoLabelManager

from ..data import data
from ..data.CustomImageDataGenerator import CustomImageDataGenerator
from ..Utils import Options
from ..model.test2 import train_model


log = logging.getLogger(__name__)

def make_viewers():
    return CustomImageDataGenerator(name="view").flow_from_indices(
        [888, 2048],
        batch_size=2,
        make_null_mask=True,
    )

class SelfTrainer:
    @staticmethod
    @Options.using_options
    def train_with_parent(
        model, parent, opt=None, train_indices=None, val_indices=None,
    ):
        if train_indices is None or val_indices is None:
            #indices = list(data.all_training_indices())
            #train_indices, val_indices = data.index_split(indices)
            end_test = data.test_range[1]
            val_indices = list(range(end_test + 1, end_test + 200))
            train_indices = list(range(end_test+200, data.n_raw_images))

        train = data.make_training_set(train_indices,
                preprocess=lambda t: t[:2448, :2048, :])
        val = data.make_validation_set(val_indices,
                preprocess=lambda t: t[:, :4096, :])
        viewers = make_viewers()

        if parent is not None:
            log.info(f"Using {parent} for pseudo data")
            train = PseudoLabelManager._generate_pseudo_data(
                    train, parent, opt.confidence_threshold)
            viewers = PseudoLabelManager._generate_pseudo_data(
                    viewers, parent, opt.confidence_threshold)

        train_model(model, train, val, viewers=viewers,)

        return model


    @staticmethod
    @Options.using_options  # rounds
    def train(model_factory, opt=None, **kwargs):
        rounds = kwargs.pop("rounds", opt.rounds)

        indices = list(data.all_training_indices())
        splitter = data.TrainValSplitManager(indices)
        train = PseudoLabelManager(data.make_training_set, splitter.train)
        val = PseudoLabelManager(data.make_validation_set, splitter.val).repeat()

        for r in range(rounds):
            log.info(f"Beginning self-training round {r+1}")
            last_round = r == rounds-1

            model = model_factory()
            model = train_model(
                model, iter(train), iter(val), round=r, summ_block=last_round, **kwargs
            )

            train._kill_current_flow()  # tensorflow is dumb and holds
            val._kill_current_flow()    # onto memory when it shouldn't

            if not last_round:
                model.trainable = False
                train.refresh(train.generate_pseudo_data, model, -0.5)

        save_loc = "/home/spencer/ohsu/research/em/final_model"
        log.info(f"Saving model at {save_loc}")
        model.save(save_loc)

        log.info("Self-training done.")

        return model

