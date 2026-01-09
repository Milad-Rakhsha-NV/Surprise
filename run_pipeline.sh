#!/bin/bash
# Complete pipeline for surprise estimation

# Configuration - modify these paths as needed
CHECKPOINT_PATH="${1:-/home/mrakhsha/Documents/Repos/IL-PhysX/logs/rsl_rl/unitree_go2_flat/2026-01-08_11-46-33/model_299.pt}"
CHECKPOINT_DIR=$(dirname "$CHECKPOINT_PATH")

echo "============================================"
echo "SURPRISE ESTIMATION PIPELINE"
echo "============================================"
echo "Checkpoint: $CHECKPOINT_PATH"
echo "Output directory: $CHECKPOINT_DIR"
echo ""

# Step 1: Collect rollout data
echo "============================================"
echo "Step 1: Collecting rollout data..."
echo "============================================"
./isaaclab.sh -p scripts/surprise_estimation/collect_rollout_data.py \
    --checkpoint "$CHECKPOINT_PATH" \
    --num_envs 64 \
    --num_steps 5000 \
    --headless

# Step 2: Train forward model
echo ""
echo "============================================"
echo "Step 2: Training forward model..."
echo "============================================"
python scripts/surprise_estimation/train_forward_model.py \
    --data_path "$CHECKPOINT_DIR/rollout_data/rollout_data.npz" \
    --num_epochs 100 \
    --batch_size 256 \
    --val_split 0.1 \
    --test_split 0.1 \
    --early_stopping 20

# Step 3: Evaluate surprise (no disturbance - baseline)
echo ""
echo "============================================"
echo "Step 3a: Evaluating surprise (no disturbance - baseline)..."
echo "============================================"
./isaaclab.sh -p scripts/surprise_estimation/evaluate_surprise_online.py \
    --checkpoint "$CHECKPOINT_PATH" \
    --forward_model "$CHECKPOINT_DIR/rollout_data/forward_model/forward_model_best.pt" \
    --num_steps 2000 \
    --num_envs 64 \
    --disturbance_type none \
    --plot \
    --headless

# Step 3b: Evaluate with external force
echo ""
echo "============================================"
echo "Step 3b: Evaluating surprise (external force 50N)..."
echo "============================================"
./isaaclab.sh -p scripts/surprise_estimation/evaluate_surprise_online.py \
    --checkpoint "$CHECKPOINT_PATH" \
    --forward_model "$CHECKPOINT_DIR/rollout_data/forward_model/forward_model_best.pt" \
    --num_steps 2000 \
    --num_envs 64 \
    --disturbance_type external_force \
    --force_magnitude 50.0 \
    --disturbance_start 500 \
    --disturbance_duration 100 \
    --plot \
    --headless

# Step 3c: Evaluate with push
echo ""
echo "============================================"
echo "Step 3c: Evaluating surprise (push velocity 1.0 m/s)..."
echo "============================================"
./isaaclab.sh -p scripts/surprise_estimation/evaluate_surprise_online.py \
    --checkpoint "$CHECKPOINT_PATH" \
    --forward_model "$CHECKPOINT_DIR/rollout_data/forward_model/forward_model_best.pt" \
    --num_steps 2000 \
    --num_envs 64 \
    --disturbance_type push \
    --push_velocity 1.0 \
    --disturbance_start 500 \
    --disturbance_duration 50 \
    --plot \
    --headless

# Step 3d: Evaluate with external torque
echo ""
echo "============================================"
echo "Step 3d: Evaluating surprise (external torque 10 Nm)..."
echo "============================================"
./isaaclab.sh -p scripts/surprise_estimation/evaluate_surprise_online.py \
    --checkpoint "$CHECKPOINT_PATH" \
    --forward_model "$CHECKPOINT_DIR/rollout_data/forward_model/forward_model_best.pt" \
    --num_steps 2000 \
    --num_envs 64 \
    --disturbance_type external_torque \
    --torque_magnitude 10.0 \
    --disturbance_start 500 \
    --disturbance_duration 100 \
    --plot \
    --headless

echo ""
echo "============================================"
echo "Pipeline complete!"
echo "Results saved in: $CHECKPOINT_DIR/surprise_evaluation/"
echo "============================================"
