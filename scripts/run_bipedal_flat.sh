#!/bin/bash
# Bipedal Go2: Rough policy on flat terrain with surprise estimation
# 
# This script runs the full pipeline for disturbance detection:
# - Uses rough-terrain trained policy
# - Evaluates on flat terrain
# - Forward model trained on flat terrain data (same distribution)
#
# Usage:
#   ./run_bipedal_flat.sh                        # defaults (16 envs, 1.5 m/s push)
#   ./run_bipedal_flat.sh --num_envs 32          # more robots
#   ./run_bipedal_flat.sh --push_velocity 2.0    # stronger push
#   ./run_bipedal_flat.sh --no_video             # skip video recording

set -e

export PATH="/home/horde/miniforge3/envs/IsaacLab/bin:$PATH"

# Model paths - these are the trained artifacts
CHECKPOINT="${CHECKPOINT:-$HOME/repos/unitree_rl_lab/logs/rsl_rl/unitree_go2_bipedal_walk_rough/0_best/best.pt}"
FORWARD_MODEL="${FORWARD_MODEL:-$HOME/repos/unitree_rl_lab/logs/rsl_rl/unitree_go2_bipedal_walk_rough/0_best/rollout_data_mixed/forward_model_flat_only/forward_model_best.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-$HOME/repos/unitree_rl_lab/logs/rsl_rl/unitree_go2_bipedal_walk_rough/0_best/surprise_evaluation_flat}"

# Flat terrain task (uses Newton backend)
TASK="Unitree-Go2-Bipedal-Walk"

# Default parameters
NUM_ENVS=16
NUM_STEPS=500
DISTURBANCE_START=200
DISTURBANCE_DURATION=50
PUSH_VELOCITY=1.5
IGNORE_WARMUP=50
VIDEO_WIDTH=1280
VIDEO_HEIGHT=720
NO_VIDEO=""

# Parse command line args
while [[ $# -gt 0 ]]; do
    case $1 in
        --num_envs) NUM_ENVS="$2"; shift 2 ;;
        --num_steps) NUM_STEPS="$2"; shift 2 ;;
        --disturbance_start) DISTURBANCE_START="$2"; shift 2 ;;
        --disturbance_duration) DISTURBANCE_DURATION="$2"; shift 2 ;;
        --push_velocity) PUSH_VELOCITY="$2"; shift 2 ;;
        --output_dir) OUTPUT_DIR="$2"; shift 2 ;;
        --checkpoint) CHECKPOINT="$2"; shift 2 ;;
        --forward_model) FORWARD_MODEL="$2"; shift 2 ;;
        --no_video) NO_VIDEO=1; shift ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --num_envs N          Number of parallel environments (default: 16)"
            echo "  --num_steps N         Simulation steps (default: 500)"
            echo "  --push_velocity V     Push velocity in m/s (default: 1.5)"
            echo "  --disturbance_start S Step to start disturbance (default: 200)"
            echo "  --disturbance_duration D Duration in steps (default: 50)"
            echo "  --output_dir DIR      Output directory"
            echo "  --checkpoint PATH     Policy checkpoint path"
            echo "  --forward_model PATH  Forward model path"
            echo "  --no_video            Skip video recording"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Verify files exist
if [[ ! -f "$CHECKPOINT" ]]; then
    echo "ERROR: Checkpoint not found: $CHECKPOINT"
    exit 1
fi
if [[ ! -f "$FORWARD_MODEL" ]]; then
    echo "ERROR: Forward model not found: $FORWARD_MODEL"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

echo "=============================================="
echo "Bipedal Go2 - Rough Policy on Flat Terrain"
echo "=============================================="
echo "Task:          $TASK"
echo "Checkpoint:    $CHECKPOINT"
echo "Forward Model: $FORWARD_MODEL"
echo "Output:        $OUTPUT_DIR"
echo ""
echo "Simulation:    $NUM_ENVS envs × $NUM_STEPS steps"
echo "Disturbance:   ${PUSH_VELOCITY} m/s push @ step ${DISTURBANCE_START} for ${DISTURBANCE_DURATION} steps"
echo "=============================================="

# Build video args
VIDEO_ARGS=""
if [[ -z "$NO_VIDEO" ]]; then
    VIDEO_ARGS="--video --video_resolution $VIDEO_WIDTH $VIDEO_HEIGHT"
fi

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="$(dirname "$SCRIPT_DIR")/src"

cd ~/repos/IsaacLab

python3 "$SRC_DIR/evaluate_surprise_online.py" \
    --task "$TASK" \
    --checkpoint "$CHECKPOINT" \
    --forward_model "$FORWARD_MODEL" \
    --output_dir "$OUTPUT_DIR" \
    --num_envs $NUM_ENVS \
    --num_steps $NUM_STEPS \
    --disturbance_type push \
    --push_velocity $PUSH_VELOCITY \
    --disturbance_start $DISTURBANCE_START \
    --disturbance_duration $DISTURBANCE_DURATION \
    --ignore_warmup_steps $IGNORE_WARMUP \
    --plot \
    $VIDEO_ARGS \
    --device cuda:0 \
    -- presets=newton

# Create synced video if video was recorded
if [[ -z "$NO_VIDEO" ]]; then
    echo ""
    echo "Creating synced video..."
    python3 "$SRC_DIR/create_synced_video.py" \
        --eval_dir "$OUTPUT_DIR" \
        --output "$OUTPUT_DIR/bipedal_flat_synced.mp4"
    
    # Compress with ffmpeg
    if command -v ffmpeg &> /dev/null; then
        ffmpeg -i "$OUTPUT_DIR/bipedal_flat_synced.mp4" \
            -c:v libx264 -preset fast -crf 23 \
            "$OUTPUT_DIR/bipedal_flat_final.mp4" -y -loglevel warning
        echo "Final video: $OUTPUT_DIR/bipedal_flat_final.mp4"
    fi
fi

echo ""
echo "=============================================="
echo "Done! Results saved to: $OUTPUT_DIR"
echo "=============================================="
