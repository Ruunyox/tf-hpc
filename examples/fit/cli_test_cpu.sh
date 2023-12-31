#! /bin/bash
#SBATCH -J tf_cli_test_cpu
#SBATCH -o ./fashionmnist/cli_test_cpu.out
#SBATCH --time=00:30:00
#SBATCH --partition=standard96:test
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=1G
#SBATCH --cpus-per-task=4

module load anaconda3/2023.09
conda activate base

export CUDA_VISIBLE_DEVICES=-1
export TF_CPP_MIN_LOG_LEVEL=2 

tfhpc --config config.yaml
