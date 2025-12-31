#!/bin/bash                                                                                                
#SBATCH --partition=TODO                     # Request partition
#SBATCH --account=TODO
#SBATCH -J explora_linprobe                  # Job name
#SBATCH -o outputs/explora_linprobe%j.out    # output file (%j expands to jobID)
#SBATCH -N 1                                 # Total number of nodes requested
#SBATCH -n 8                                 # Total number of cores requested
#SBATCH --get-user-env                       # retrieve the users login environment
#SBATCH --mem=90000                          # server cpu memory requested (per node)
#SBATCH -t 48:00:00                          # Time limit (hh:mm:ss)
#SBATCH --gres=gpu:1                         # Type/number of GPUs needed
#SBATCH --constraint=16G                     # gpu memory

# If not using SLURM, set num_gpus manually
if [ -n "$SLURM_JOB_GPUS" ]; then
  num_gpus=$(echo $SLURM_JOB_GPUS | tr ',' '\n' | wc -l)
else
  num_gpus=1
fi

# list out some useful information (optional)
echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
echo "SLURM_NNODES"=$SLURM_NNODES
echo "SLURM_GPUS"=$SLURM_JOB_GPUS

echo "working directory = "$SLURM_SUBMIT_DIR

# ============ Paths ============
base_dir="data_and_checkpoints"
base_csv_dir="${base_dir}/fmow_csvs"
base_experiment_dir="${base_dir}/explora_linprobe"

# ============ Config ============
# cfg_file="linprobe/configs/fmow_vitb14.yaml"
cfg_file="linprobe/configs/fmow_vitl14.yaml"
# cfg_file="linprobe/configs/fmow_vitg14.yaml"

# ============ Pretrained weights ============
# Vanilla DINOv2 pretrained weights. If using this, set the cfg_file to be linprobe/configs/vitl14_pretrain.yaml
# pretrained_weights="${base_dir}/dinov2_vitl14_pretrain.pth"

# ExPLoRA pretrained weights
pretrained_weights="${base_dir}/explora_dinov2_vit_large_fmow_rgb_encoder_only.pth"

# ============ Output directory ============
out_dir="${base_experiment_dir}/linprobe-explora_dino-rgb-blk23r64-bs256-epochs10"

# ============ Run ============
export PYTHONPATH=.
WANDB__SERVICE_WAIT=300 torchrun --nproc_per_node=$num_gpus --master_port=40001 -m linprobe.linear \
    --wandb=explora_linprobe \
    --wandb_entity=TODO \
    --train-dataset="fmow:data_and_checkpoints/fmow_csvs/train_62classes.csv" \
    --val-dataset="fmow:data_and_checkpoints/fmow_csvs/test_62classes.csv" \
    --config-file="${cfg_file}" \
    --output-dir="${out_dir}" \
    --pretrained-weights="${pretrained_weights}" \
    --batch-size=256 \
    --epochs=10 \
    --epoch-length=1500 \
    --save-checkpoint-frequency=1500 \
    --eval-period-iterations=1500
