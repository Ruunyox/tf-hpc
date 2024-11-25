import argparse
import tensorflow as tf
import tensorflow_datasets as tfds
from jsonargparse import CLI, ArgumentParser, ActionConfigFile
from ..nn.models import *
from typing import List, Optional, Any, Union
import os
import absl.logging

absl.logging.set_verbosity(absl.logging.ERROR)

strategies = {"mirrored_strategy": tf.distribute.MirroredStrategy, None: None}
cross_device_ops = {
    "hierarchical_copy_all_reduce": tf.distribute.HierarchicalCopyAllReduce,
    "nccl_all_reduce": tf.distribute.NcclAllReduce,
    "reduction_to_one_device": tf.distribute.ReductionToOneDevice,
    None: None,
}
cast_types = {"float32": tf.float32}


def prepare_dataset(cfg: Dict) -> Tuple:
    if not cfg.cache_data:
        rc = tfds.ReadConfig(try_autocache=False)
        cfg.dataset["read_config"] = rc
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

    if cfg.cache_data:
        train_dataset = train_dataset.cache()
        if val_dataset is not None:
            val_dataset = val_dataset.cache()
    return train_dataset, val_dataset


def main():
    parser = ArgumentParser()
    parser.add_class_arguments(ModelWrapper, "model_wrapper")
    parser.add_argument("--cache_data", type=bool, default=False)
    parser.add_argument("--strategy.name", type=Optional[str])
    parser.add_argument("--x_cast", type=Optional[str])
    parser.add_argument("--y_cast", type=Optional[str])
    parser.add_argument("--min_log_level", type=Optional[Union[str, int]])
    parser.add_argument("--strategy.opts.devices", type=Optional[List[str]])
    parser.add_argument("--strategy.opts.cross_device_ops.op", type=Optional[str])
    parser.add_argument("--strategy.opts.cross_device_ops.opts", type=Optional[Any])
    parser.add_function_arguments(CompileBuilder, "compile", fail_untyped=False)
    parser.add_function_arguments(FitBuilder, "fit", fail_untyped=False)
    parser.add_function_arguments(tfds.load, "dataset")
    parser.add_argument("--config", action=ActionConfigFile)

    cfg = parser.parse_args()

    if cfg.min_log_level is not None:
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = str(cfg.min_log_level)

    if cfg.strategy is not None:
        if cfg.strategy.opts.cross_device_ops.opts is None:
            cross_device_ops_opts = {}
        else:
            cross_device_ops_opts = cfg.strategy.opts.cross_device_ops.opts
        # Strategy setup must be handled BEFORE JSONArgparse class instantiation
        strategy = strategies[cfg.strategy.name](
            cross_device_ops=cross_device_ops[cfg.strategy.opts.cross_device_ops.op](
                **cross_device_ops_opts
            )
            if cfg.strategy.opts.cross_device_ops.op is not None
            else None
        )
        with strategy.scope():
            cfg = parser.instantiate_classes(cfg)
            compile_opts = CompileBuilder(**cfg.compile)
            fit_opts = FitBuilder(**cfg.fit)

            train_dataset, val_dataset = prepare_dataset(cfg)
            cfg.model_wrapper.model.compile(**compile_opts)
            tf.config.list_physical_devices()
            print(
                f"Using strategy {strategy} with devices {strategy._extended._devices}"
            )
            cfg.model_wrapper.model.fit(
                x=train_dataset, validation_data=val_dataset, **fit_opts
            )

    else:
        cfg = parser.instantiate_classes(cfg)
        compile_opts = CompileBuilder(**cfg.compile)
        fit_opts = FitBuilder(**cfg.fit)

        train_dataset, val_dataset = prepare_dataset(cfg)
        cfg.model_wrapper.model.compile(**compile_opts)

        cfg.model_wrapper.model.fit(
            x=train_dataset, validation_data=val_dataset, **fit_opts
        )


if __name__ == "__main__":
    main()
