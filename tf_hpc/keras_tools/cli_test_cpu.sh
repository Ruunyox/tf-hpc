#! /bin/bash
#SBATCH -J cli_test_cpu
#SBATCH -o cli_test_cpu.out
#SBATCH --time=00:30:00
#SBATCH --partition=standard96:test
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=1G
#SBATCH --cpus-per-task=4

# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/home/bzfbnick/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/home/bzfbnick/miniconda3/etc/profile.d/conda.sh" ]; then
        . "/home/bzfbnick/miniconda3/etc/profile.d/conda.sh"
    else
        export PATH="/home/bzfbnick/miniconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<

conda activate tf_test

python train.py --config config.yaml
