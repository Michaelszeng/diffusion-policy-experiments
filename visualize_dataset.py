"""
Visualize trajectory videos from any zarr dataset in this repository.
RGB keys are detected automatically by finding 4D arrays (T, H, W, C) in data/.
A separate window (or output file) is created for each camera.

CURRENTLY ONLY WORKS FOR ZARR DATASETS

Display videos interactively
python visualize_dataset.py \
    --zarr-path /home/michzeng/diffusion-policy/data/diffusion_experiments/planar_pushing/sim_sim_tee_data_carbon_large.zarr \
    --fps 30 \
    --stride 1 \
    --image-size 640x480

Save videos to disk at 2x speed
python visualize_dataset.py \
    --zarr-path ~/home/michzeng/diffusion-policy/data/diffusion_experiments/planar_pushing/sim_sim_tee_data_carbon_large.zarr \
    --fps 30 \
    --stride 1 \
    --image-size 640x480 \
    --output-path ./videos \
    --playback-speed 2.0

Only show specific cameras
python visualize_dataset.py \
    --zarr-path data/diffusion_experiments/planar_pushing/sim_sim_tee_data_carbon_large.zarr \
    --fps 30 \
    --cameras overhead_camera wrist_camera
"""

import argparse
import os

import cv2
import numpy as np
import zarr


def find_rgb_keys(dataset):
    """Return keys under data/ whose arrays are 4D (T, H, W, C) — i.e. image arrays."""
    data_group = dataset["data"]
    rgb_keys = []
    for key in data_group.keys():
        arr = data_group[key]
        if arr.ndim == 4 and arr.shape[-1] in (1, 3, 4):
            rgb_keys.append(key)
    return sorted(rgb_keys)


def play_trajectory_videos(
    zarr_path,
    fps,
    stride=1,
    image_size=None,
    output_path=None,
    playback_speed=1.0,
    cameras=None,
):
    """
    Play trajectory videos from all RGB cameras in a zarr dataset.

    Args:
        zarr_path (str): Path to the zarr dataset.
        fps (int): Frames per second for the video.
        stride (int): Step size over episodes (1 = every episode).
        image_size (tuple): Desired (width, height) for resizing. None = original size.
        output_path (str): Directory to save videos. None = display only.
            Videos are saved as output_path/<camera_key>/trajectory_N.mp4.
        playback_speed (float): Speed multiplier. 1.0 = normal, 2.0 = 2x speed.
        cameras (list[str]): Explicit list of camera keys to use. None = auto-detect.
    """
    dataset = zarr.open(zarr_path, mode="r")
    episode_ends = dataset["meta"]["episode_ends"][:]
    episode_starts = [0] + episode_ends[:-1].tolist()

    if cameras is not None:
        rgb_keys = cameras
        for key in rgb_keys:
            if key not in dataset["data"]:
                raise ValueError(f"Camera key '{key}' not found in data/. "
                                 f"Available keys: {list(dataset['data'].keys())}")
    else:
        rgb_keys = find_rgb_keys(dataset)

    if not rgb_keys:
        raise RuntimeError(
            f"No 4D image arrays found in data/. "
            f"Available keys: {list(dataset['data'].keys())}"
        )

    print(f"Found {len(rgb_keys)} camera(s): {rgb_keys}")
    print(f"Dataset has {len(episode_starts)} episodes.")

    if output_path is not None:
        for key in rgb_keys:
            os.makedirs(os.path.join(output_path, key), exist_ok=True)

    delay_ms = max(1, int(1000 / (fps * playback_speed)))
    adjusted_fps = fps * playback_speed

    for i in range(0, len(episode_starts), stride):
        start_idx = episode_starts[i]
        end_idx = episode_ends[i]

        # Load frames for every camera upfront so windows stay in sync
        camera_frames = {}
        for key in rgb_keys:
            raw = dataset["data"][key][start_idx:end_idx]
            frames = np.array([cv2.cvtColor(img, cv2.COLOR_RGB2BGR) for img in raw])
            if image_size is not None:
                frames = np.array([cv2.resize(img, image_size) for img in frames])
            camera_frames[key] = frames

        if output_path is not None:
            for key, frames in camera_frames.items():
                video_file = os.path.join(output_path, key, f"trajectory_{i}.mp4")
                h, w = frames[0].shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(video_file, fourcc, adjusted_fps, (w, h))
                for frame in frames:
                    writer.write(frame)
                writer.release()
                print(f"Saved {video_file}")
        else:
            n_frames = min(len(f) for f in camera_frames.values())
            print(f"Episode {i + 1}/{len(episode_starts)}  ({n_frames} frames)  — press 'q' to quit")
            for t in range(n_frames):
                for key, frames in camera_frames.items():
                    cv2.imshow(f"Episode {i + 1} | {key}", frames[t])
                if cv2.waitKey(delay_ms) & 0xFF == ord("q"):
                    print("Interrupted by user.")
                    cv2.destroyAllWindows()
                    return

            for key in rgb_keys:
                cv2.destroyWindow(f"Episode {i + 1} | {key}")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize trajectory videos from any zarr dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--zarr-path", type=str, required=True, help="Path to the zarr dataset.")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second.")
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Step size over episodes (default: 1 = every episode).",
    )
    parser.add_argument(
        "--image-size",
        type=lambda s: tuple(map(int, s.split("x"))),
        default=None,
        help="Resize images to WIDTHxHEIGHT (e.g. 640x480). Default: original size.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Directory to save videos. If omitted, videos are displayed interactively.",
    )
    parser.add_argument(
        "--playback-speed",
        type=float,
        default=1.0,
        help="Speed multiplier (1.0 = normal, 2.0 = 2x, 0.5 = half speed).",
    )
    parser.add_argument(
        "--cameras",
        type=str,
        nargs="+",
        default=None,
        help="Camera keys to display (e.g. --cameras overhead_camera wrist_camera). "
             "Default: auto-detect all 4D arrays in data/.",
    )
    args = parser.parse_args()

    play_trajectory_videos(
        zarr_path=args.zarr_path,
        fps=args.fps,
        stride=args.stride,
        image_size=args.image_size,
        output_path=args.output_path,
        playback_speed=args.playback_speed,
        cameras=args.cameras,
    )
