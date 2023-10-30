import argparse
import tensorflow as tf
import tensorflow_datasets as tfds
from jsonargparse import CLI, ArgumentParser, ActionConfigFile
from tf_hpc.nn.models import *


def main():
    parser = ArgumentParser()
    parser.add_class_arguments(ModelWrapper, "model_wrapper")
    parser.add_function_arguments(CompileBuilder, "compile", fail_untyped=False)
    parser.add_function_arguments(FitBuilder, "fit", fail_untyped=False)
    parser.add_function_arguments(tfds.load, "dataset")
    parser.add_argument("--config", action=ActionConfigFile)

    cfg = parser.parse_args()
    cfg = parser.instantiate_classes(cfg)

    dataset = tfds.load(**cfg.dataset)
    compile_opts = CompileBuilder(**cfg.compile)
    fit_opts = FitBuilder(**cfg.fit)

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
    cfg.model_wrapper.model.compile(**compile_opts)
    cfg.model_wrapper.model.fit(
        x=train_dataset, validation_data=val_dataset, **fit_opts
    )


if __name__ == "__main__":
    main()
