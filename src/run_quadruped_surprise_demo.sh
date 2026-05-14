#!/bin/bash
# Run quadruped surprise evaluation with flat terrain and generate synced video
# Usage: ./run_quadruped_surprise_demo.sh [push_velocity]
#
# Prerequisites:
#   - IsaacLab environment with unitree_rl_lab installed
#   - Trained quadruped policy and forward model
#
# Example:
#   ./run_quadruped_surprise_demo.sh 0.5    # Push at 0.5 m/s
#   ./run_quadruped_surprise_demo.sh 1.0    # Push at 1.0 m/s

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SURPRISE_REPO="$(dirname "$SCRIPT_DIR")"
UNITREE_REPO="$HOME/repos/unitree_rl_lab"
ISAACLAB_DIR="$HOME/repos/IsaacLab"

# Ensure conda env is active
export PATH="$HOME/miniforge3/envs/IsaacLab/bin:$PATH"

# Parameters
PUSH_VELOCITY="${1:-0.5}"  # Default 0.5 m/s
NUM_ENVS=9
NUM_STEPS=500
CHECKPOINT="$UNITREE_REPO/logs/rsl_rl/unitree_go2_velocity_rough/0_best/best.pt"
FORWARD_MODEL="$UNITREE_REPO/logs/rsl_rl/unitree_go2_velocity_rough/0_best/rollout_data/forward_model/forward_model_best.pt"
OUTPUT_DIR="$UNITREE_REPO/logs/rsl_rl/unitree_go2_velocity_rough/0_best/surprise_evaluation"

echo "=============================================="
echo "Quadruped Surprise Evaluation Demo"
echo "=============================================="
echo "Push velocity: $PUSH_VELOCITY m/s"
echo "Num envs: $NUM_ENVS"
echo "Num steps: $NUM_STEPS"
echo "Output dir: $OUTPUT_DIR"
echo ""

# Step 1: Run evaluation with video recording
echo "[1/2] Running surprise evaluation..."
cd "$ISAACLAB_DIR"
./isaaclab.sh -p "$SCRIPT_DIR/evaluate_surprise_online.py" \
    --task Unitree-Go2-Velocity-Rough \
    --checkpoint "$CHECKPOINT" \
    --forward_model "$FORWARD_MODEL" \
    --num_envs "$NUM_ENVS" \
    --disturbance_type push \
    --push_velocity "$PUSH_VELOCITY" \
    --video \
    --plot \
    --num_steps "$NUM_STEPS" \
    --flat_terrain

# Step 2: Create synced video
echo ""
echo "[2/2] Creating synchronized video..."
OUTPUT_VIDEO="$OUTPUT_DIR/quadruped_flat_push${PUSH_VELOCITY}_synced.mp4"

python3 "$SCRIPT_DIR/create_synced_video.py" \
    --eval_dir "$OUTPUT_DIR" \
    --output "$OUTPUT_VIDEO" \
    --layout side_by_side

echo ""
echo "=============================================="
echo "Done!"
echo "Output video: $OUTPUT_VIDEO"
echo "=============================================="
