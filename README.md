# Tensorflow-HPC

============================

### Tensorflow testing suite for HPC environment/module development
-------------------------------

Simple tests to check that Tensorflow utilities work for HPC deployment. For rapid
testing, Keras model building/training is controlled through `jsonargparse` YAML
configuration files.

### Usage
-------------------------------

Any `tensorflow-datasets` builtin dataset can be used with the configuration YAML.
Different example YAMLs and SLURM submission scripts for CPU and DDP-GPU training are included in
`examples`.

After installation (`pip install .`), run the following:

`tfhpc --config PATH_TO_YOUR_CONFIG`

To train a model. Example configs and SLURM submission scripts can be found in
the `examples` folder. Be aware that you may need to predownload/cache
`tensorflow_datasets` datasets using login nodes if cluster nodes have no active
internet connection. By default, this occurs in `$HOME/tensorflow_datasets`.
