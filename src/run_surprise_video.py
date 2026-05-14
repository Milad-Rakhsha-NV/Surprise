#!/usr/bin/env python3
"""Run surprise evaluation with video recording and create synchronized overlay.

This script:
1. Runs the surprise evaluation with optional push disturbance
2. Records video of the simulation
3. Creates an animated timeline graph synced to the video
4. Combines them into a final overlay video

Usage:
    # Using isaaclab.sh wrapper (recommended):
    ./isaaclab.sh -p scripts/surprise_estimation/run_surprise_video.py \
        --task Unitree-Go2-Bipedal-Walk \
        --checkpoint logs/rsl_rl/unitree_go2_bipedal_walk_rough/0_best/best.pt \
        --forward_model logs/rsl_rl/unitree_go2_bipedal_walk_rough/0_best/rollout_data/forward_model/forward_model_best.pt \
        --num_envs 9 \
        --disturbance_type push \
        --push_velocity 0.5 \
        --output_dir ./surprise_videos

    # Direct python (requires IsaacLab environment):
    python run_surprise_video.py [same args]
"""

import argparse
import os
import sys
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import cv2


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run surprise evaluation with video and create synced overlay."
    )
    
    # Task and model paths
    parser.add_argument("--task", type=str, default="Unitree-Go2-Bipedal-Walk",
                        help="Task name (default: Unitree-Go2-Bipedal-Walk for flat terrain)")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to policy checkpoint (.pt)")
    parser.add_argument("--forward_model", type=str, required=True,
                        help="Path to trained forward model (.pt)")
    
    # Simulation settings
    parser.add_argument("--num_envs", type=int, default=9,
                        help="Number of parallel environments (default: 9 for 3x3 grid)")
    parser.add_argument("--num_steps", type=int, default=500,
                        help="Number of simulation steps (default: 500 = 10s at 50Hz)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    
    # Disturbance settings
    parser.add_argument("--disturbance_type", type=str, default="push",
                        choices=["none", "push", "external_force", "external_torque"],
                        help="Type of disturbance (default: push)")
    parser.add_argument("--push_velocity", type=float, default=0.5,
                        help="Push velocity in m/s (default: 0.5, gentle push)")
    parser.add_argument("--force_magnitude", type=float, default=20.0,
                        help="External force magnitude in N")
    parser.add_argument("--torque_magnitude", type=float, default=5.0,
                        help="External torque magnitude in Nm")
    parser.add_argument("--disturbance_start", type=int, default=200,
                        help="Step at which disturbance starts (default: 200 = 4s)")
    parser.add_argument("--disturbance_duration", type=int, default=20,
                        help="Duration of disturbance in steps (default: 20 = 0.4s)")
    
    # Video settings
    parser.add_argument("--video_resolution", type=int, nargs=2, default=[1280, 720],
                        metavar=('WIDTH', 'HEIGHT'),
                        help="Video resolution (default: 1280 720)")
    parser.add_argument("--fps", type=int, default=50,
                        help="Output video FPS (default: 50, matches simulation)")
    parser.add_argument("--layout", type=str, default="side_by_side",
                        choices=["side_by_side", "stacked"],
                        help="Video layout (default: side_by_side)")
    
    # Output settings
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: auto-generated)")
    parser.add_argument("--output_name", type=str, default=None,
                        help="Output video name (default: auto-generated)")
    
    # Flags
    parser.add_argument("--keep_intermediate", action="store_true",
                        help="Keep intermediate files (raw video, npz, plots)")
    parser.add_argument("--skip_eval", action="store_true",
                        help="Skip evaluation, use existing files in output_dir")
    
    return parser.parse_args()


def run_evaluation(args, eval_output_dir: str) -> tuple[str, str]:
    """Run the surprise evaluation script.
    
    Returns:
        Tuple of (video_path, results_path)
    """
    print("\n" + "="*70)
    print("STEP 1: Running Surprise Evaluation")
    print("="*70)
    
    # Build the evaluation command
    # We need to call the evaluate_surprise_online.py script
    script_dir = Path(__file__).parent
    eval_script = script_dir / "evaluate_surprise_online.py"
    
    cmd = [
        sys.executable, str(eval_script),
        "--task", args.task,
        "--checkpoint", args.checkpoint,
        "--forward_model", args.forward_model,
        "--num_envs", str(args.num_envs),
        "--num_steps", str(args.num_steps),
        "--seed", str(args.seed),
        "--disturbance_type", args.disturbance_type,
        "--push_velocity", str(args.push_velocity),
        "--force_magnitude", str(args.force_magnitude),
        "--torque_magnitude", str(args.torque_magnitude),
        "--disturbance_start", str(args.disturbance_start),
        "--disturbance_duration", str(args.disturbance_duration),
        "--video_resolution", str(args.video_resolution[0]), str(args.video_resolution[1]),
        "--video",
        "--plot",
        "--output_dir", eval_output_dir,
    ]
    
    print(f"[INFO]: Running: {' '.join(cmd[:5])}...")
    
    # Run the evaluation
    result = subprocess.run(cmd, capture_output=False)
    
    if result.returncode != 0:
        raise RuntimeError(f"Evaluation failed with return code {result.returncode}")
    
    # Find the output files
    video_dir = Path(eval_output_dir) / "videos"
    videos = list(video_dir.glob("*.mp4"))
    if not videos:
        raise FileNotFoundError(f"No video found in {video_dir}")
    video_path = str(max(videos, key=lambda x: x.stat().st_mtime))
    
    results = list(Path(eval_output_dir).glob("surprise_results_*.npz"))
    if not results:
        raise FileNotFoundError(f"No results found in {eval_output_dir}")
    results_path = str(max(results, key=lambda x: x.stat().st_mtime))
    
    print(f"[INFO]: Video: {video_path}")
    print(f"[INFO]: Results: {results_path}")
    
    return video_path, results_path


def create_synced_video(
    video_path: str,
    results_path: str,
    output_path: str,
    layout: str = "side_by_side",
    fps: int = 50,
):
    """Create synchronized video with simulation and animated timeline graph.
    
    This follows the same approach as the G1/H1/Go2 create_synced_video.py script.
    """
    print("\n" + "="*70)
    print("STEP 2: Creating Synchronized Video Overlay")
    print("="*70)
    
    # Load results
    print(f"[INFO]: Loading results from {results_path}")
    data = np.load(results_path)
    
    surprise_per_env = data["surprise_per_env"]
    pred_error_per_env = data["pred_error_per_env"]
    disturbance_start = int(data["disturbance_start"])
    disturbance_end = int(data["disturbance_end"])
    
    num_steps, num_envs = surprise_per_env.shape
    dt = 0.02  # 50Hz simulation
    timestamps = np.arange(num_steps) * dt
    
    # Compute statistics
    mean_surprise = surprise_per_env.mean(axis=1)
    std_surprise = surprise_per_env.std(axis=1)
    mean_error = pred_error_per_env.mean(axis=1)
    
    dist_start_time = disturbance_start * dt
    dist_end_time = disturbance_end * dt
    
    print(f"[INFO]: Data: {num_steps} steps, {timestamps[-1]:.2f}s, {num_envs} envs")
    print(f"[INFO]: Disturbance: {dist_start_time:.2f}s to {dist_end_time:.2f}s")
    print(f"[INFO]: Surprise range: {mean_surprise.min():.0f} to {mean_surprise.max():.0f}")
    
    # Load video
    print(f"[INFO]: Loading video from {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration = total_video_frames / video_fps
    
    print(f"[INFO]: Video: {video_width}x{video_height} @ {video_fps:.1f}fps, {video_duration:.2f}s")
    
    # Setup figure for graphs (high quality, matching the original style)
    fig_height = 7.2
    fig_width = 12
    dpi = 100
    
    fig, axes = plt.subplots(2, 1, figsize=(fig_width, fig_height), dpi=dpi)
    fig.patch.set_facecolor('white')
    
    # Top plot: Surprise with confidence band
    ax1 = axes[0]
    ax1.fill_between(timestamps, mean_surprise - std_surprise, mean_surprise + std_surprise,
                     alpha=0.3, color='blue', label='±1 Std across envs')
    line1, = ax1.plot(timestamps, mean_surprise, 'b-', linewidth=1.5, label='Mean surprise')
    
    # Disturbance window
    if dist_start_time > 0:
        ax1.axvspan(dist_start_time, dist_end_time, alpha=0.3, color='red', label='Push Applied')
        ax1.axvline(x=dist_start_time, color='red', linestyle='--', linewidth=2)
    
    ax1.set_ylabel('Surprise (NLL sum)', fontsize=11)
    ax1.set_title(f'Online Surprise Estimation ({num_envs} robots)', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, timestamps[-1])
    
    # Dynamic y-limits with padding
    y_min = mean_surprise.min() - abs(mean_surprise.min()) * 0.1 - 500
    y_max = mean_surprise.max() + abs(mean_surprise.max()) * 0.1 + 500
    ax1.set_ylim(y_min, y_max)
    
    # Bottom plot: Prediction Error
    ax2 = axes[1]
    ax2.plot(timestamps, mean_error, 'g-', linewidth=1.5, label='Mean prediction error')
    if dist_start_time > 0:
        ax2.axvspan(dist_start_time, dist_end_time, alpha=0.3, color='red')
        ax2.axvline(x=dist_start_time, color='red', linestyle='--', linewidth=2)
    ax2.set_xlabel('Time (seconds)', fontsize=11)
    ax2.set_ylabel('Prediction Error (MSE)', fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, timestamps[-1])
    ax2.legend(loc='upper left', fontsize=9)
    
    plt.tight_layout()
    
    # Create markers for current time - large and visible
    marker1, = ax1.plot([], [], 'o', markersize=14, color='yellow',
                        markeredgecolor='red', markeredgewidth=2.5, zorder=10)
    marker2, = ax2.plot([], [], 'o', markersize=14, color='yellow',
                        markeredgecolor='red', markeredgewidth=2.5, zorder=10)
    
    # Vertical line for current time
    vline1 = ax1.axvline(x=0, color='green', linewidth=2, alpha=0.8)
    vline2 = ax2.axvline(x=0, color='green', linewidth=2, alpha=0.8)
    
    # Time text
    time_text = ax1.text(0.02, 0.95, '', transform=ax1.transAxes, fontsize=13,
                         verticalalignment='top', fontweight='bold',
                         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))
    
    # Status text
    status_text = ax1.text(0.98, 0.95, '', transform=ax1.transAxes, fontsize=12,
                           verticalalignment='top', horizontalalignment='right',
                           fontweight='bold',
                           bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.9))
    
    # Surprise value text
    surprise_text = ax1.text(0.02, 0.78, '', transform=ax1.transAxes, fontsize=11,
                             verticalalignment='top',
                             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    
    def fig_to_array(fig):
        """Convert matplotlib figure to numpy array."""
        fig.canvas.draw()
        buf = fig.canvas.buffer_rgba()
        img = np.asarray(buf)
        return cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    
    # Calculate output dimensions
    graph_width = int(fig_width * dpi)
    graph_height = int(fig_height * dpi)
    
    if layout == "side_by_side":
        scale = graph_height / video_height
        scaled_video_width = int(video_width * scale)
        scaled_video_height = graph_height
        output_width = scaled_video_width + graph_width
        output_height = graph_height
    else:  # stacked
        output_width = max(video_width, graph_width)
        output_height = video_height + graph_height
    
    # Setup output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (output_width, output_height))
    
    if not out.isOpened():
        # Try H264
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        out = cv2.VideoWriter(output_path, fourcc, fps, (output_width, output_height))
    
    if not out.isOpened():
        raise RuntimeError("Could not open video writer")
    
    print(f"[INFO]: Output: {output_width}x{output_height} @ {fps}fps")
    print(f"[INFO]: Generating synchronized video...")
    
    # Generate frames
    num_output_frames = int(video_duration * fps)
    
    for frame_idx in range(num_output_frames):
        current_time = frame_idx / fps
        
        # Find corresponding data index
        data_idx = min(int(current_time / dt), num_steps - 1)
        
        # Update markers
        marker1.set_data([current_time], [mean_surprise[data_idx]])
        marker2.set_data([current_time], [mean_error[data_idx]])
        vline1.set_xdata([current_time, current_time])
        vline2.set_xdata([current_time, current_time])
        
        # Update time text
        time_text.set_text(f't = {current_time:.2f}s')
        
        # Update status
        if dist_start_time <= current_time <= dist_end_time:
            status_text.set_text('⚠️ PUSH ACTIVE')
            status_text.set_bbox(dict(boxstyle='round', facecolor='red', alpha=0.9))
        elif current_time > dist_end_time and current_time < dist_end_time + 2.0:
            status_text.set_text('POST-DISTURBANCE')
            status_text.set_bbox(dict(boxstyle='round', facecolor='orange', alpha=0.9))
        else:
            status_text.set_text('NOMINAL')
            status_text.set_bbox(dict(boxstyle='round', facecolor='lightgreen', alpha=0.9))
        
        # Update surprise value
        surprise_text.set_text(f'Surprise: {mean_surprise[data_idx]:.0f}')
        
        # Render graph
        graph_frame = fig_to_array(fig)
        
        # Get video frame
        video_frame_idx = int(current_time * video_fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, video_frame_idx)
        ret, video_frame = cap.read()
        
        if not ret:
            video_frame = np.zeros((video_height, video_width, 3), dtype=np.uint8)
        
        # Combine frames
        if layout == "side_by_side":
            video_resized = cv2.resize(video_frame, (scaled_video_width, scaled_video_height))
            combined = np.hstack([video_resized, graph_frame])
        else:
            video_resized = cv2.resize(video_frame, (output_width, video_height))
            graph_resized = cv2.resize(graph_frame, (output_width, graph_height))
            combined = np.vstack([video_resized, graph_resized])
        
        out.write(combined)
        
        if frame_idx % 100 == 0:
            pct = 100 * frame_idx / num_output_frames
            print(f"  Frame {frame_idx}/{num_output_frames} ({pct:.1f}%)")
    
    cap.release()
    out.release()
    plt.close(fig)
    
    print(f"[INFO]: ✅ Video saved to {output_path}")


def compress_video(input_path: str, output_path: str):
    """Compress video using ffmpeg for smaller file size."""
    print("\n[INFO]: Compressing video with ffmpeg...")
    
    cmd = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-c:v", "libx264",
        "-crf", "23",
        "-preset", "medium",
        "-movflags", "+faststart",
        output_path
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        # Get file sizes
        input_size = os.path.getsize(input_path) / 1024 / 1024
        output_size = os.path.getsize(output_path) / 1024 / 1024
        print(f"[INFO]: Compressed {input_size:.1f}MB -> {output_size:.1f}MB")
        return True
    else:
        print(f"[WARNING]: ffmpeg compression failed, keeping original")
        return False


def main():
    args = parse_args()
    
    # Setup output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_dir = Path(args.checkpoint).parent
        output_dir = checkpoint_dir / f"surprise_video_{timestamp}"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("SURPRISE EVALUATION WITH VIDEO OVERLAY")
    print("="*70)
    print(f"Task:            {args.task}")
    print(f"Checkpoint:      {args.checkpoint}")
    print(f"Forward model:   {args.forward_model}")
    print(f"Num envs:        {args.num_envs}")
    print(f"Num steps:       {args.num_steps} ({args.num_steps * 0.02:.1f}s)")
    print(f"Disturbance:     {args.disturbance_type}")
    if args.disturbance_type == "push":
        print(f"Push velocity:   {args.push_velocity} m/s")
    print(f"Output dir:      {output_dir}")
    print("="*70)
    
    # Step 1: Run evaluation (unless skipped)
    eval_output_dir = str(output_dir / "eval")
    
    if args.skip_eval:
        print("\n[INFO]: Skipping evaluation, using existing files...")
        video_dir = Path(eval_output_dir) / "videos"
        videos = list(video_dir.glob("*.mp4"))
        video_path = str(max(videos, key=lambda x: x.stat().st_mtime))
        results = list(Path(eval_output_dir).glob("surprise_results_*.npz"))
        results_path = str(max(results, key=lambda x: x.stat().st_mtime))
    else:
        video_path, results_path = run_evaluation(args, eval_output_dir)
    
    # Step 2: Create synchronized video
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output_name:
        output_name = args.output_name
    else:
        task_short = args.task.replace("Unitree-", "").replace("-", "_").lower()
        output_name = f"surprise_{task_short}_{args.disturbance_type}_{timestamp}"
    
    temp_output = str(output_dir / f"{output_name}_raw.mp4")
    final_output = str(output_dir / f"{output_name}.mp4")
    
    create_synced_video(
        video_path=video_path,
        results_path=results_path,
        output_path=temp_output,
        layout=args.layout,
        fps=args.fps,
    )
    
    # Step 3: Compress with ffmpeg
    if compress_video(temp_output, final_output):
        if not args.keep_intermediate:
            os.remove(temp_output)
    else:
        # Keep raw if compression failed
        os.rename(temp_output, final_output)
    
    # Print summary
    print("\n" + "="*70)
    print("✅ COMPLETE!")
    print("="*70)
    print(f"Final video: {final_output}")
    print(f"File size:   {os.path.getsize(final_output) / 1024 / 1024:.1f} MB")
    print("="*70)


if __name__ == "__main__":
    main()
