#!/usr/bin/env python3
"""Create a synchronized video combining simulation footage with animated surprise graphs.

This script overlays the surprise metrics visualization on top of or alongside the
simulation video, with a moving marker showing the current time position.
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle
import cv2
from datetime import datetime


def load_results(results_path: str) -> dict:
    """Load surprise evaluation results."""
    data = np.load(results_path, allow_pickle=True)
    
    surprise = data["surprise_per_env"]
    num_steps = surprise.shape[0]
    dt = 0.02  # 50Hz simulation
    timestamps = np.arange(num_steps) * dt
    
    return {
        "surprise": surprise,
        "ema_surprise": data["surprise_ema_per_env"],
        "prediction_error": data["pred_error_per_env"],
        "disturbance_active": data["disturbance_active"],
        "timestamps": timestamps,
        "disturbance_start": int(data["disturbance_start"]),
        "disturbance_end": int(data["disturbance_end"]),
    }


def create_synced_video(
    video_path: str,
    results_path: str,
    output_path: str,
    layout: str = "side_by_side",
    graph_fps: int = 50,
):
    """Create synchronized video with simulation and animated graphs.
    
    Args:
        video_path: Path to the simulation video.
        results_path: Path to the .npz results file.
        output_path: Path for the output video.
        layout: 'side_by_side' or 'overlay'.
        graph_fps: FPS for the output video.
    """
    # Load data
    print(f"[INFO]: Loading results from {results_path}")
    results = load_results(results_path)
    
    surprise = results["surprise"]
    prediction_error = results["prediction_error"]
    disturbance_active = results["disturbance_active"]
    timestamps = results["timestamps"]
    
    num_steps, num_envs = surprise.shape
    
    # Compute statistics
    mean_surprise = surprise.mean(axis=1)
    std_surprise = surprise.std(axis=1)
    mean_error = prediction_error.mean(axis=1)
    
    # Find disturbance window
    dist_start_idx = results["disturbance_start"]
    dist_end_idx = results["disturbance_end"]
    dist_start_time = timestamps[dist_start_idx] if dist_start_idx < num_steps else 0
    dist_end_time = timestamps[min(dist_end_idx, num_steps-1)] if dist_end_idx > 0 else 0
    
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
    
    print(f"[INFO]: Video: {video_width}x{video_height} @ {video_fps}fps, {total_video_frames} frames, {video_duration:.2f}s")
    print(f"[INFO]: Data: {num_steps} steps, {timestamps[-1]:.2f}s")
    
    # Setup figure for graphs (high quality)
    fig_height = 7.2  # Match video height ratio
    fig_width = 12
    dpi = 150  # Higher DPI for better quality
    
    fig, axes = plt.subplots(2, 1, figsize=(fig_width, fig_height), dpi=dpi)
    fig.patch.set_facecolor('white')
    
    # Top plot: Surprise
    ax1 = axes[0]
    ax1.fill_between(timestamps, mean_surprise - std_surprise, mean_surprise + std_surprise,
                     alpha=0.3, color='blue', label='Std across envs')
    ax1.plot(timestamps, mean_surprise, 'b-', linewidth=1.5, label='Mean surprise')
    
    # Add disturbance window
    if dist_start_time > 0:
        ax1.axvspan(dist_start_time, dist_end_time, alpha=0.3, color='red', label='Disturbance')
        ax1.axvline(x=dist_start_time, color='red', linestyle='--', linewidth=2)
    
    ax1.set_ylabel('Surprise (NLL sum)', fontsize=11)
    ax1.set_title(f'Online Surprise with Time Marker ({num_envs} envs)', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, timestamps[-1])
    
    # Bottom plot: MSE
    ax2 = axes[1]
    ax2.plot(timestamps, mean_error, 'g-', linewidth=1.5)
    if dist_start_time > 0:
        ax2.axvspan(dist_start_time, dist_end_time, alpha=0.3, color='red')
        ax2.axvline(x=dist_start_time, color='red', linestyle='--', linewidth=2)
    ax2.set_xlabel('Time (seconds)', fontsize=11)
    ax2.set_ylabel('Prediction Error (MSE)', fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, timestamps[-1])
    
    plt.tight_layout()
    
    # Create markers (circles) for current time - larger and more visible
    marker1, = ax1.plot([], [], 'o', markersize=16, color='yellow', 
                        markeredgecolor='red', markeredgewidth=3, zorder=10)
    marker2, = ax2.plot([], [], 'o', markersize=16, color='yellow',
                        markeredgecolor='red', markeredgewidth=3, zorder=10)
    
    # Vertical line for current time
    vline1 = ax1.axvline(x=0, color='red', linewidth=2.5, alpha=0.8)
    vline2 = ax2.axvline(x=0, color='red', linewidth=2.5, alpha=0.8)
    
    # Time text - larger font
    time_text = ax1.text(0.02, 0.95, '', transform=ax1.transAxes, fontsize=14,
                         verticalalignment='top', fontweight='bold',
                         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))
    
    # Status text - larger and more prominent
    status_text = ax1.text(0.98, 0.95, '', transform=ax1.transAxes, fontsize=13,
                           verticalalignment='top', horizontalalignment='right',
                           fontweight='bold',
                           bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.9))
    
    # Surprise value text
    surprise_text = ax1.text(0.02, 0.75, '', transform=ax1.transAxes, fontsize=12,
                             verticalalignment='top', fontweight='bold',
                             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    
    # Convert figure to image
    def fig_to_array(fig):
        fig.canvas.draw()
        buf = fig.canvas.buffer_rgba()
        img = np.asarray(buf)
        return cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    
    # Calculate output dimensions
    graph_width = int(fig_width * dpi)
    graph_height = int(fig_height * dpi)
    
    if layout == "side_by_side":
        # Scale video to match graph height
        scale = graph_height / video_height
        scaled_video_width = int(video_width * scale)
        scaled_video_height = graph_height
        output_width = scaled_video_width + graph_width
        output_height = graph_height
    else:  # overlay or stacked
        output_width = max(video_width, graph_width)
        output_height = video_height + graph_height
    
    # Setup output video (use H264 for better quality)
    # Try different codecs in order of preference
    codecs = ['avc1', 'H264', 'X264', 'mp4v']
    out = None
    for codec in codecs:
        fourcc = cv2.VideoWriter_fourcc(*codec)
        out = cv2.VideoWriter(output_path, fourcc, graph_fps, (output_width, output_height))
        if out.isOpened():
            print(f"[INFO]: Using codec: {codec}")
            break
        out.release()
    
    if not out.isOpened():
        raise RuntimeError("Could not open video writer with any codec")
    
    print(f"[INFO]: Output video: {output_width}x{output_height} @ {graph_fps}fps")
    print(f"[INFO]: Creating synchronized video...")
    
    # Generate frames
    num_output_frames = int(video_duration * graph_fps)
    
    for frame_idx in range(num_output_frames):
        current_time = frame_idx / graph_fps
        
        # Find corresponding data index
        data_idx = np.searchsorted(timestamps, current_time)
        data_idx = min(data_idx, num_steps - 1)
        
        # Update markers
        marker1.set_data([current_time], [mean_surprise[data_idx]])
        marker2.set_data([current_time], [mean_error[data_idx]])
        vline1.set_xdata([current_time, current_time])
        vline2.set_xdata([current_time, current_time])
        
        # Update time text
        time_text.set_text(f't = {current_time:.2f}s')
        
        # Update status
        if dist_start_time <= current_time <= dist_end_time:
            status_text.set_text('DISTURBANCE ACTIVE')
            status_text.set_bbox(dict(boxstyle='round', facecolor='red', alpha=0.9))
        elif current_time > dist_end_time and current_time < dist_end_time + 1.5:
            status_text.set_text('POST-DISTURBANCE')
            status_text.set_bbox(dict(boxstyle='round', facecolor='orange', alpha=0.9))
        else:
            status_text.set_text('NOMINAL')
            status_text.set_bbox(dict(boxstyle='round', facecolor='lightgreen', alpha=0.9))
        
        # Update surprise value display
        surprise_text.set_text(f'Surprise: {mean_surprise[data_idx]:.1f}')
        
        # Render graph
        graph_frame = fig_to_array(fig)
        
        # Get video frame
        video_frame_idx = int(current_time * video_fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, video_frame_idx)
        ret, video_frame = cap.read()
        
        if not ret:
            # Use last valid frame or black frame
            video_frame = np.zeros((video_height, video_width, 3), dtype=np.uint8)
        
        # Combine frames
        if layout == "side_by_side":
            # Resize video
            video_resized = cv2.resize(video_frame, (scaled_video_width, scaled_video_height))
            # Combine
            combined = np.hstack([video_resized, graph_frame])
        else:
            # Stack vertically
            video_resized = cv2.resize(video_frame, (output_width, video_height))
            graph_resized = cv2.resize(graph_frame, (output_width, graph_height))
            combined = np.vstack([video_resized, graph_resized])
        
        out.write(combined)
        
        if frame_idx % 50 == 0:
            print(f"  Frame {frame_idx}/{num_output_frames} ({100*frame_idx/num_output_frames:.1f}%)")
    
    cap.release()
    out.release()
    plt.close(fig)
    
    print(f"[INFO]: Video saved to {output_path}")


def find_latest_files(eval_dir: str):
    """Find the latest video and results files."""
    video_dir = os.path.join(eval_dir, "videos")
    
    # Find latest video
    videos = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]
    if not videos:
        raise FileNotFoundError(f"No videos found in {video_dir}")
    videos.sort(key=lambda x: os.path.getmtime(os.path.join(video_dir, x)), reverse=True)
    latest_video = os.path.join(video_dir, videos[0])
    
    # Find latest results
    results = [f for f in os.listdir(eval_dir) if f.startswith('surprise_results') and f.endswith('.npz')]
    if not results:
        raise FileNotFoundError(f"No results found in {eval_dir}")
    results.sort(key=lambda x: os.path.getmtime(os.path.join(eval_dir, x)), reverse=True)
    latest_results = os.path.join(eval_dir, results[0])
    
    return latest_video, latest_results


def main():
    parser = argparse.ArgumentParser(description="Create synchronized video with simulation and graphs.")
    parser.add_argument("--video", type=str, default=None, help="Path to simulation video.")
    parser.add_argument("--results", type=str, default=None, help="Path to results .npz file.")
    parser.add_argument("--eval_dir", type=str, default=None, 
                        help="Evaluation directory (auto-finds latest video and results).")
    parser.add_argument("--output", type=str, default=None, help="Output video path.")
    parser.add_argument("--layout", type=str, default="side_by_side",
                        choices=["side_by_side", "stacked"], help="Video layout.")
    parser.add_argument("--fps", type=int, default=50, help="Output video FPS (default matches simulation).")
    
    args = parser.parse_args()
    
    # Find files
    if args.eval_dir:
        video_path, results_path = find_latest_files(args.eval_dir)
        print(f"[INFO]: Found video: {video_path}")
        print(f"[INFO]: Found results: {results_path}")
    else:
        video_path = args.video
        results_path = args.results
        
    if not video_path or not results_path:
        raise ValueError("Must provide --video and --results, or --eval_dir")
    
    # Output path
    if args.output:
        output_path = args.output
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.dirname(results_path)
        output_path = os.path.join(output_dir, f"synced_video_{timestamp}.mp4")
    
    create_synced_video(
        video_path=video_path,
        results_path=results_path,
        output_path=output_path,
        layout=args.layout,
        graph_fps=args.fps,
    )


if __name__ == "__main__":
    main()

