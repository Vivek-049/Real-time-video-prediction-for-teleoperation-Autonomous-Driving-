#!/bin/bash
# ============================================================================
# Training Monitor Script for IFRNet t+3 Training
# Usage: ./check_training.sh
# ============================================================================

echo "=============================================="
echo "  IFRNet t+3 Training Monitor"
echo "=============================================="
echo ""

# Check if tmux session is running
echo "=== TMUX Session Status ==="
if tmux has-session -t ifrvp_train 2>/dev/null; then
    echo "✓ Training session 'ifrvp_train' is RUNNING"
else
    echo "✗ Training session 'ifrvp_train' NOT FOUND"
    echo "  Training may have completed or not started yet."
fi
echo ""

# Check GPU status
echo "=== GPU Status ==="
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw --format=csv,noheader,nounits | while read line; do
    IFS=',' read -r idx name util mem_used mem_total temp power <<< "$line"
    echo "GPU $idx: ${util}% util | ${mem_used}/${mem_total} MB | ${temp}°C | ${power}W"
done
echo ""

# Find latest log directory
LOG_BASE="/storage_drive_0/vivek/video/IFRVP/IFRVP/checkpoint_bdd100k_t3/IFRNet"
if [ -d "$LOG_BASE" ]; then
    LATEST_LOG=$(ls -td "$LOG_BASE"/*/ 2>/dev/null | head -1)
    if [ -n "$LATEST_LOG" ]; then
        echo "=== Training Log (last 20 lines) ==="
        echo "Log dir: $LATEST_LOG"
        echo ""
        if [ -f "${LATEST_LOG}train.log" ]; then
            tail -20 "${LATEST_LOG}train.log"
        else
            echo "Waiting for training to start logging..."
        fi
        echo ""
        
        # Check for saved checkpoints
        echo "=== Saved Checkpoints ==="
        ls -lh "${LATEST_LOG}"*.pth 2>/dev/null || echo "No checkpoints saved yet"
        echo ""
        
        # Parse progress from log
        echo "=== Training Progress ==="
        if [ -f "${LATEST_LOG}train.log" ]; then
            # Get last epoch info
            LAST_EPOCH=$(grep -oP 'epoch:\K[0-9]+/[0-9]+' "${LATEST_LOG}train.log" | tail -1)
            LAST_LOSS=$(grep -oP 'loss_rec:\K[0-9.e+-]+' "${LATEST_LOG}train.log" | tail -1)
            if [ -n "$LAST_EPOCH" ]; then
                echo "Current: Epoch $LAST_EPOCH"
                echo "Loss (rec): $LAST_LOSS"
            fi
        fi
    else
        echo "No training logs found yet in $LOG_BASE"
    fi
else
    echo "Log directory not created yet: $LOG_BASE"
    echo "Training may not have started."
fi
echo ""

# Show running time
echo "=== Process Info ==="
ps aux | grep "train_bdd100k_t3.py" | grep -v grep | head -3
echo ""

echo "=============================================="
echo "  To view live training: tmux attach -t ifrvp_train"
echo "  To detach from tmux: Ctrl+B then D"
echo "=============================================="







