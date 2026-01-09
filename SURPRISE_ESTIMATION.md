# Surprise Estimation for Anomaly Detection in RL Policies

This document describes the mathematical foundations, implementation details, and usage of the surprise estimation system for detecting anomalies in reinforcement learning policies.

## 1. Overview

The goal is to compute a real-time "surprise" signal that indicates when the robot's behavior deviates from what it learned during training. This can be used for:
- Detecting hardware faults
- Identifying out-of-distribution situations
- Triggering soft shutdowns before failures
- Monitoring policy performance in deployment

## 2. Mathematical Foundation

### 2.1 Problem Setup

Consider a robot with:
- **Observation space**: $o \in \mathbb{R}^{d_o}$ (joint positions, velocities, IMU, etc.)
- **Action space**: $a \in \mathbb{R}^{d_a}$ (joint commands)
- **Trained policy**: $\pi(a | o)$

During nominal operation, the robot follows predictable dynamics:

$$o_{t+1} = f(o_t, a_t) + \epsilon$$

where $f$ is the unknown true dynamics and $\epsilon$ is process noise.

### 2.2 Forward Model

We learn a probabilistic forward model that predicts the next observation given the current state and action:

$$p_\theta(o_{t+1} | o_t, a_t) = \mathcal{N}\big(o_{t+1}; \mu_\theta(o_t, a_t), \Sigma_\theta(o_t, a_t)\big)$$

We use a diagonal covariance for computational efficiency:

$$\Sigma_\theta = \text{diag}(\sigma_1^2, \sigma_2^2, \ldots, \sigma_{d_o}^2)$$

The neural network outputs:
- **Mean**: $\mu_\theta \in \mathbb{R}^{d_o}$
- **Log-variance**: $\log \sigma^2_\theta \in \mathbb{R}^{d_o}$ (log for numerical stability)

### 2.3 Surprise as Negative Log-Likelihood

Surprise at time $t$ is defined as the negative log-likelihood of the observed next state:

$$S_t = -\log p_\theta(o_{t+1} | o_t, a_t)$$

For a Gaussian distribution:

$$S_t = \frac{1}{2} \sum_{i=1}^{d_o} \left( \log \sigma_i^2 + \frac{(o_{t+1,i} - \mu_i)^2}{\sigma_i^2} \right) + \text{const}$$

**Intuition**: 
- If the actual $o_{t+1}$ is close to predicted $\mu$: low surprise
- If the actual $o_{t+1}$ is far from predicted $\mu$: high surprise
- If the model is confident (small $\sigma^2$): deviations penalized more
- If the model is uncertain (large $\sigma^2$): deviations tolerated more

### 2.4 Why Learn Variance?

Learning the variance $\sigma^2$ is critical because:

1. **Heteroscedastic noise**: Different observations may have different noise levels (joint velocity is noisier than joint position)

2. **Calibrated uncertainty**: The model learns which predictions to trust

3. **Proper scoring rule**: NLL with learned variance encourages the model to output well-calibrated uncertainty estimates

Without learned variance (fixed $\sigma^2 = 1$):
- MSE becomes the loss
- All observation dimensions treated equally
- No calibrated uncertainty

## 3. Training the Forward Model

### 3.1 Data Collection

We collect nominal transition data by rolling out the trained policy:

$$\mathcal{D} = \{(o_t^{(i)}, a_t^{(i)}, o_{t+1}^{(i)})\}_{i=1}^N$$

Important considerations:
- Only collect from **nominal operation** (no failures)
- Filter out transitions where episode terminated (reset invalidates next observation)
- Collect diverse states by running many parallel environments

### 3.2 Normalization

All inputs are normalized using statistics from the training data:

$$\tilde{o} = \frac{o - \mu_o}{\sigma_o}, \quad \tilde{a} = \frac{a - \mu_a}{\sigma_a}$$

This is essential because:
- Different observation components have different scales
- Neural networks train better with normalized inputs
- The same normalization must be used during inference

### 3.3 Loss Function

The training objective is to minimize the average negative log-likelihood:

$$\mathcal{L}(\theta) = \frac{1}{N} \sum_{i=1}^N -\log p_\theta(o_{t+1}^{(i)} | o_t^{(i)}, a_t^{(i)})$$

Expanding:

$$\mathcal{L}(\theta) = \frac{1}{2N} \sum_{i=1}^N \sum_{j=1}^{d_o} \left( \log \sigma_j^2(\theta) + \frac{(o_{t+1,j}^{(i)} - \mu_j(\theta))^2}{\sigma_j^2(\theta)} \right)$$

### 3.4 Network Architecture

```
Input: [o_t, a_t] ∈ ℝ^(d_o + d_a)
    │
    ├─ Linear(d_o + d_a, 256)
    ├─ LayerNorm(256)
    ├─ ReLU
    │
    ├─ Linear(256, 256)
    ├─ LayerNorm(256)
    ├─ ReLU
    │
    ├─ Linear(256, 256)
    ├─ LayerNorm(256)
    ├─ ReLU
    │
    ├─────────────────────────────┐
    │                             │
    ▼                             ▼
Linear(256, d_o)           Linear(256, d_o)
    │                             │
    ▼                             ▼
  μ_θ                        log σ²_θ
(mean)                    (log-variance)
```

**Design choices**:
- LayerNorm instead of BatchNorm (works with any batch size)
- Separate heads for mean and variance (allows different learning dynamics)
- Log-variance output clamped to $[-10, 2]$ for numerical stability
- Log-variance head initialized to output small variance ($\approx 0.135$)

### 3.5 Training Procedure

1. **Data split**: Train (80%) / Validation (10%) / Test (10%)
2. **Optimizer**: AdamW with weight decay $10^{-4}$
3. **Learning rate schedule**: Cosine annealing
4. **Gradient clipping**: Max norm 1.0
5. **Early stopping**: Stop if validation loss doesn't improve for 20 epochs
6. **Best model selection**: Save model with lowest validation NLL

### 3.6 Convergence Indicators

The model is converging well if:
- Training and validation loss both decrease
- Validation loss stabilizes (not increasing = no overfitting)
- Generalization gap (val - train) stays small and stable
- Per-dimension MSE is reasonable across all observation components

**Note**: NLL can be negative when $\sigma^2 < 1$, which is expected for normalized data with good predictions.

Warning signs:
- Validation loss increasing while training loss decreases → overfitting
- Very large generalization gap → need more data or regularization
- Some dimensions have much higher MSE → potential data quality issues

## 4. Online Surprise Estimation

### 4.1 Inference Pipeline

At each timestep:
1. Normalize current observation and action
2. Forward pass through trained model to get $(\mu, \log \sigma^2)$
3. Observe actual next observation
4. Normalize next observation
5. Compute surprise:

$$S_t = \frac{1}{2} \sum_{i=1}^{d_o} \left( \log \sigma_i^2 + \frac{(\tilde{o}_{t+1,i} - \mu_i)^2}{\sigma_i^2} \right)$$

### 4.2 Exponential Moving Average (EMA)

Raw surprise can be noisy. We smooth with EMA:

$$\bar{S}_t = \alpha \cdot S_t + (1 - \alpha) \cdot \bar{S}_{t-1}$$

Typical $\alpha \in [0.05, 0.2]$.

### 4.3 Thresholding

Set a threshold based on nominal surprise statistics:

$$\text{threshold} = \mu_S + k \cdot \sigma_S$$

where $\mu_S, \sigma_S$ are mean and std of surprise during nominal operation.

Typical $k \in [3, 6]$ (depends on false positive tolerance).

### 4.4 Actions

When surprise exceeds threshold:
- **Warning level**: Log event, reduce gains
- **Critical level**: Trigger soft shutdown

## 5. Physical Disturbance Testing

### 5.1 Supported Disturbance Types

| Type | Description | Typical Magnitude |
|------|-------------|-------------------|
| `external_force` | Apply force to robot base | 20-100 N |
| `external_torque` | Apply torque to robot base | 5-20 Nm |
| `push` | Instantaneous velocity change | 0.5-2.0 m/s |

### 5.2 Expected Behavior

When a disturbance is applied:
1. Robot experiences unexpected dynamics
2. Actual $o_{t+1}$ differs from predicted $\mu$
3. Surprise increases
4. After disturbance ends, surprise should return to baseline

### 5.3 Validation Criteria

The surprise estimator is working correctly if:
- Baseline surprise is stable and low
- Surprise increases significantly during disturbance
- Increase factor is typically 2-10x depending on disturbance magnitude
- All environments show similar response

## 6. Implementation Files

| File | Purpose |
|------|---------|
| `collect_rollout_data.py` | Collect $(o, a, o')$ transitions from trained policy |
| `train_forward_model.py` | Train probabilistic forward model with NLL loss |
| `evaluate_surprise_online.py` | Online surprise computation with physical disturbances |
| `analyze_results.py` | Post-hoc analysis and threshold recommendation |

## 7. Usage Example

### Step 1: Collect Data
```bash
./isaaclab.sh -p scripts/surprise_estimation/collect_rollout_data.py \
    --checkpoint logs/rsl_rl/unitree_go2_flat/2026-01-08_11-46-33/model_299.pt \
    --num_envs 64 \
    --num_steps 5000 \
    --headless
```

### Step 2: Train Forward Model
```bash
python scripts/surprise_estimation/train_forward_model.py \
    --data_path logs/rsl_rl/unitree_go2_flat/2026-01-08_11-46-33/rollout_data/rollout_data.npz \
    --num_epochs 100 \
    --batch_size 256
```

### Step 3: Evaluate with External Force
```bash
./isaaclab.sh -p scripts/surprise_estimation/evaluate_surprise_online.py \
    --checkpoint logs/rsl_rl/unitree_go2_flat/2026-01-08_11-46-33/model_299.pt \
    --forward_model logs/rsl_rl/unitree_go2_flat/2026-01-08_11-46-33/rollout_data/forward_model/forward_model_best.pt \
    --disturbance_type external_force \
    --force_magnitude 50.0 \
    --disturbance_start 500 \
    --disturbance_duration 100 \
    --num_envs 64 \
    --plot \
    --headless
```

### Step 4: Evaluate with Push
```bash
./isaaclab.sh -p scripts/surprise_estimation/evaluate_surprise_online.py \
    --checkpoint logs/rsl_rl/unitree_go2_flat/2026-01-08_11-46-33/model_299.pt \
    --forward_model logs/rsl_rl/unitree_go2_flat/2026-01-08_11-46-33/rollout_data/forward_model/forward_model_best.pt \
    --disturbance_type push \
    --push_velocity 1.0 \
    --disturbance_start 500 \
    --disturbance_duration 50 \
    --num_envs 64 \
    --plot \
    --headless
```

## 8. Observation Space Reference (Go2)

The Go2 flat environment observation vector (48 dimensions):

| Index | Component | Dimensions | Description |
|-------|-----------|------------|-------------|
| 0-2 | base_lin_vel | 3 | Base linear velocity (body frame) |
| 3-5 | base_ang_vel | 3 | Base angular velocity (body frame) |
| 6-8 | projected_gravity | 3 | Gravity projected to body frame |
| 9-11 | velocity_commands | 3 | Target velocity commands |
| 12-23 | joint_pos | 12 | Joint positions relative to default |
| 24-35 | joint_vel | 12 | Joint velocities |
| 36-47 | actions | 12 | Previous action |

## 9. Mathematical Connection to Active Inference

This implementation is conceptually equivalent to the "prediction error" in active inference:

| Active Inference | This Implementation |
|-----------------|---------------------|
| Generative model | Forward model $p_\theta(o_{t+1}|o_t, a_t)$ |
| Sensory prediction | Model output $\mu_\theta$ |
| Precision | Inverse variance $1/\sigma^2$ |
| Free energy | Negative log-likelihood (surprise) |
| Prediction error | $o_{t+1} - \mu_\theta$ |

The key difference is we don't do belief updating over latent states - we directly use observations. This is simpler and sufficient for anomaly detection.

## 10. Limitations and Extensions

### Current Limitations
- Assumes single-step prediction (no recurrent state)
- Diagonal covariance (ignores cross-dimension correlations)
- Does not model transition distribution (only observation distribution)

### Potential Extensions
- **Ensemble models**: Train multiple forward models, use disagreement as additional uncertainty
- **Recurrent models**: LSTM/Transformer for temporal dependencies
- **Latent space models**: VAE-based prediction in latent space
- **Full covariance**: Learn full covariance matrix for better uncertainty
