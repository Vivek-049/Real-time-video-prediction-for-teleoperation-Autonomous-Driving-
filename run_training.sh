#!/bin/bash
# =============================================================================
# BDD100K t+3 Prediction Training Pipeline
# =============================================================================
# 
# This script runs the complete training pipeline:
# 1. Preprocess BDD100K videos into t+3 triplets
# 2. Train IFRNet on the triplets
#
# Prerequisites:
#   - Conda environment 'ifrvp' with PyTorch installed
#   - BDD100K videos extracted in /storage_drive_0/vivek/video/datasets/
#
# =============================================================================

set -e  # Exit on error

# Configuration
CONDA_PATH="/storage_drive_0/vivek/miniconda3"
ENV_NAME="ifrvp"
WORK_DIR="/storage_drive_0/vivek/video/IFRVP"
VIDEO_DIR="/storage_drive_0/vivek/video/datasets/bdd100k_videos_train_00/bdd100k/videos/train"
TRIPLET_DIR="/storage_drive_0/vivek/video/datasets/bdd100k_t3_triplets"

# Training config
MODEL_NAME="IFRNet"  # Options: IFRNet, IFRNet_S, IFRNet_L
TARGET_SIZE=512
BATCH_SIZE=6  # Per GPU
NUM_GPUS=2
EPOCHS=100
PRETRAINED_PATH="IFRVP/IFRVP_k+1_laploss/IFRNet_S_latest.pth"  # Set to "" to train from scratch

# Activate conda
source ${CONDA_PATH}/etc/profile.d/conda.sh
conda activate ${ENV_NAME}

cd ${WORK_DIR}

echo "=============================================="
echo "BDD100K t+3 Prediction Training Pipeline"
echo "=============================================="
echo ""
echo "Configuration:"
echo "  Model: ${MODEL_NAME}"
echo "  Target size: ${TARGET_SIZE}x${TARGET_SIZE}"
echo "  Batch size: ${BATCH_SIZE} per GPU x ${NUM_GPUS} GPUs"
echo "  Epochs: ${EPOCHS}"
echo "  Pretrained: ${PRETRAINED_PATH:-None (training from scratch)}"
echo ""

# =============================================================================
# Step 1: Preprocess videos into triplets (skip if already done)
# =============================================================================
if [ ! -f "${TRIPLET_DIR}/tri_trainlist.txt" ]; then
    echo "=============================================="
    echo "Step 1: Preprocessing videos into t+3 triplets"
    echo "=============================================="
    python preprocess_bdd100k_t3.py \
        --input_dir "${VIDEO_DIR}" \
        --output_dir "${TRIPLET_DIR}" \
        --size ${TARGET_SIZE} \
        --skip_frames 5 \
        --num_workers 8
else
    echo "=============================================="
    echo "Step 1: Skipping preprocessing (triplets already exist)"
    echo "=============================================="
    echo "  Triplet dir: ${TRIPLET_DIR}"
    TRIPLET_COUNT=$(wc -l < "${TRIPLET_DIR}/tri_trainlist.txt")
    echo "  Triplets: ${TRIPLET_COUNT}"
fi

echo ""

# =============================================================================
# Step 2: Train model
# =============================================================================
echo "=============================================="
echo "Step 2: Training ${MODEL_NAME} on t+3 triplets"
echo "=============================================="

cd IFRVP

# Build training command
TRAIN_CMD="torchrun --nproc_per_node=${NUM_GPUS} train_bdd100k_t3.py \
    --model_name ${MODEL_NAME} \
    --dataset_dir ${TRIPLET_DIR} \
    --batch_size ${BATCH_SIZE} \
    --crop_size ${TARGET_SIZE} \
    --epochs ${EPOCHS} \
    --eval_interval 5 \
    --world_size ${NUM_GPUS}"

# Add pretrained path if specified
if [ -n "${PRETRAINED_PATH}" ] && [ -f "${WORK_DIR}/${PRETRAINED_PATH}" ]; then
    TRAIN_CMD="${TRAIN_CMD} --pretrained_path ${WORK_DIR}/${PRETRAINED_PATH}"
    echo "Fine-tuning from: ${PRETRAINED_PATH}"
else
    echo "Training from scratch"
fi

echo ""
echo "Running: ${TRAIN_CMD}"
echo ""

eval ${TRAIN_CMD}

echo ""
echo "=============================================="
echo "Training complete!"
echo "=============================================="
echo "Checkpoints saved to: IFRVP/checkpoint_bdd100k_t3/${MODEL_NAME}/"

