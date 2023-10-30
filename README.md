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
