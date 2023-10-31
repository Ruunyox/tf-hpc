import argparse
import tensorflow as tf
import tensorflow_datasets as tfds
from jsonargparse import CLI, ArgumentParser, ActionConfigFile
from ..nn.models import *
from typing import List, Optional

strategies = {"mirrored_strategy": tf.distribute.MirroredStrategy}


def main():
    parser = ArgumentParser()
    parser.add_class_arguments(ModelWrapper, "model_wrapper")
    parser.add_argument("--strategy.name", type=Optional[str])
    parser.add_argument("--strategy.opts.devices", type=Optional[List[str]])
    parser.add_argument(
        "--strategy.opts.cross_device_ops", type=Optional[tf.distribute.CrossDeviceOps]
    )
    parser.add_function_arguments(CompileBuilder, "compile", fail_untyped=False)
    parser.add_function_arguments(FitBuilder, "fit", fail_untyped=False)
    parser.add_function_arguments(tfds.load, "dataset")
    parser.add_argument("--config", action=ActionConfigFile)

    cfg = parser.parse_args()
    if cfg.strategy is not None:
        strategy = strategies[cfg.strategy.name](**cfg.strategy.opts)
        with strategy.scope():
            cfg = parser.instantiate_classes(cfg)
            compile_opts = CompileBuilder(**cfg.compile)
            fit_opts = FitBuilder(**cfg.fit)
            cfg.model_wrapper.model.compile(**compile_opts)
    else:
        cfg = parser.instantiate_classes(cfg)
        compile_opts = CompileBuilder(**cfg.compile)
        fit_opts = FitBuilder(**cfg.fit)
        cfg.model_wrapper.model.compile(**compile_opts)

    dataset = tfds.load(**cfg.dataset)
    if isinstance(dataset, list):
        if len(dataset) not in [1, 2]:
            raise RuntimeError("Dataset splitting only supports two splits")
        else:
            if len(dataset) == 1:
                train_dataset = dataset[0]
                val_dataset = None
            if len(dataset) == 2:
                train_dataset = dataset[0]
                val_dataset = dataset[1]
    elif isinstance(dataset, dict):
        train_dataset = dataset["train"]
        val_dataset = None

    cfg.model_wrapper.model.fit(
        x=train_dataset, validation_data=val_dataset, **fit_opts
    )


if __name__ == "__main__":
    main()
