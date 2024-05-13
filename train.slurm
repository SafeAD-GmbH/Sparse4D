#!/bin/bash
#SBATCH --job-name=safead-sparse-hz-lr2e4
#SBATCH --gpus=8
#SBATCH --ntasks-per-node=8                     # Must match number of GPUs
#SBATCH --nice=200
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=3G
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --requeue
#SBATCH --qos=normal

export TMP_CLONE_DIR=$HOME"/tmp/deletes_every_fortnight/dir/"$SLURM_JOBID

if [ -d "$TMP_CLONE_DIR" ]; then
   echo "'$TMP_CLONE_DIR' found continuing the experiment there"
else
   echo "'$TMP_CLONE_DIR' not found -> copying to this directory"
   cp -r --reflink=always . $TMP_CLONE_DIR
fi

cd $TMP_CLONE_DIR

export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export WORLD_SIZE=$(($SLURM_JOB_NUM_NODES * $SLURM_NTASKS_PER_NODE))
echo "WORLD_SIZE="$WORLD_SIZE

export OMP_NUM_THREADS=1

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

DIR=$(dirname "$0")
export PYTHONPATH="$DIR":$PYTHONPATH

srun python tools/train.py --deterministic --launcher=slurm $@
