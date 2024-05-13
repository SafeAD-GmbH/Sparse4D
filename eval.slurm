#!/bin/bash
#SBATCH --job-name=bevformer_eval
#SBATCH --gpus=4
#SBATCH --ntasks-per-node=4                     # Must match number of GPUs
#SBATCH --nice=200
#SBATCH --nodes=1
#SBATCH --cpus-per-task=6
#SBATCH --mem-per-cpu=4G
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --requeue
#SBATCH --qos=normal

export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export WORLD_SIZE=$(($SLURM_JOB_NUM_NODES * $SLURM_NTASKS_PER_NODE))
echo "WORLD_SIZE="$WORLD_SIZE

export OMP_NUM_THREADS=1

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

DIR=$(dirname "$0")
export PYTHONPATH="$DIR":$PYTHONPATH

srun python tools/test.py --eval bbox --launcher=slurm $@
