#!/bin/bash
#SBATCH --job-name=jupyter-h200
#SBATCH --account=sfglab
#SBATCH --partition=hopper-2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=500G
#SBATCH --gres=gpu:h200:1
#SBATCH --time=24:00:00
#SBATCH --output=/mnt/evafs/groups/sfglab/ksengupta/pchic/logs/jupyter_%j.log

source /mnt/evafs/groups/sfglab/ksengupta/soft/miniforge3/etc/profile.d/conda.sh
conda activate pchic

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMBA_NUM_THREADS=$SLURM_CPUS_PER_TASK

port=$((20000 + SLURM_JOB_ID % 20000))
node=$(hostname -s)
user=$(whoami)

echo "====================================================="
echo "Node: ${node}   Port: ${port}   GPU: ${CUDA_VISIBLE_DEVICES}"
echo "On your laptop run:"
echo "  ssh -N -L ${port}:${node}:${port} -J <ssh_login>@ssh.mini.pw.edu.pl ${user}@eden.mini.pw.edu.pl"
echo "Then open in browser: http://localhost:${port}/lab?token=... (token below)"
echo "====================================================="

nvidia-smi --query-gpu=name,memory.total --format=csv

cd /mnt/evafs/groups/sfglab/ksengupta/pchic/pic_hic_ml
jupyter lab --no-browser --ip=0.0.0.0 --port=${port}
