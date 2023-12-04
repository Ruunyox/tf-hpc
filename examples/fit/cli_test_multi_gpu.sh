#! /bin/bash
#SBATCH -J tf_cli_test_multi_gpu
#SBATCH -o ./fashionmnist_multi_gpu/cli_test_multi_gpu.out
#SBATCH --time=00:30:00
#SBATCH --partition=gpu-a100
#SBATCH --reservation=a100_tests
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:A100:4
#SBATCH --mem-per-cpu=1G
#SBATCH --cpus-per-task=4

module load sw.a100
module load nvhpc/23.1 
module load cuda/11.8
module load anaconda3/2023.09

conda activate base

export TF_CPP_MIN_LOG_LEVEL=2
export XLA_FLAGS=--xla_gpu_cuda_data_dir=/sw/compiler/cuda/11.8/a100/install 

tfhpc --config config_multi_gpu.yaml
