"""
Pre-process Le2i dataset: decode each AVI once and save frames + audio to disk.

Usage:
    python3 -m spwm.data.preprocess_le2i --data_root Le2i --out_root Le2i_processed
"""
import os, sys, argparse, time
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm


def preprocess_video(video_path: str, out_dir: str, data_root: str, frame_size: int = 224) -> bool:
    """Decode a single AVI and save frames.pt + audio.pt."""
    from av import open as av_open

    # Include scene name to avoid collisions across scenes
    rel = os.path.relpath(video_path, data_root)
    vid_name = rel.replace('/', '_').replace('\\', '_').replace('.avi', '').replace('.AVI', '')
    vid_out = os.path.join(out_dir, vid_name)
    os.makedirs(vid_out, exist_ok=True)

    frames_path = os.path.join(vid_out, 'frames.pt')
    audio_path = os.path.join(vid_out, 'audio.pt')
    meta_path = os.path.join(vid_out, 'meta.pt')

    if os.path.exists(meta_path):
        return True  # already processed

    try:
        frames = []
        audio_samples = []

        with av_open(video_path) as container:
            video_stream = container.streams.video[0]
            fps = float(video_stream.average_rate) if video_stream.average_rate else 25.0

            for frame in container.decode(video=0):
                img = frame.to_ndarray(format='rgb24')
                frames.append(torch.from_numpy(img).permute(2, 0, 1).float() / 255.0)

            try:
                audio_stream = container.streams.audio[0]
                audio_sr = audio_stream.rate or 48000
                for frame in container.decode(audio=0):
                    samples = frame.to_ndarray()
                    audio_samples.append(torch.from_numpy(samples.astype(np.float32)))
            except Exception:
                audio_sr = 48000

        # Stack and resize frames
        frames_tensor = torch.stack(frames) if frames else torch.zeros(1, 3, frame_size, frame_size)

        if frames_tensor.shape[-1] != frame_size:
            from torch.nn.functional import interpolate
            T, C, H, W = frames_tensor.shape
            frames_tensor = frames_tensor.reshape(T * C, H, W).unsqueeze(0)
            frames_tensor = interpolate(
                frames_tensor, size=(frame_size, frame_size),
                mode='bilinear', align_corners=False,
            ).squeeze(0).reshape(T, C, frame_size, frame_size)

        # Save frames
        torch.save(frames_tensor, frames_path)

        # Stack audio
        if audio_samples:
            audio_tensor = torch.cat(audio_samples, dim=-1).mean(dim=0)
        else:
            audio_tensor = torch.zeros(1)
        torch.save(audio_tensor, audio_path)

        # Meta
        torch.save({'fps': fps, 'n_frames': len(frames), 'audio_sr': audio_sr}, meta_path)

        return True

    except Exception as e:
        print(f"  ERROR {vid_name}: {e}")
        return False


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data_root', default='Le2i')
    p.add_argument('--out_root', default='Le2i_processed')
    p.add_argument('--frame_size', type=int, default=224)
    args = p.parse_args()

    # Find all videos
    video_paths = []
    import glob
    for pat in ['*.avi', '*.AVI', '*.mp4', '*.MP4']:
        video_paths.extend(glob.glob(os.path.join(args.data_root, '**', pat), recursive=True))

    print(f"Found {len(video_paths)} videos")
    out_dir = args.out_root
    os.makedirs(out_dir, exist_ok=True)

    t0 = time.time()
    n_ok, n_fail, n_skip = 0, 0, 0
    for vp in tqdm(video_paths, desc="Pre-processing"):
        # Check if already processed (using scene-prefixed name)
        rel = os.path.relpath(vp, args.data_root)
        vid_name = rel.replace('/', '_').replace('\\', '_').replace('.avi', '').replace('.AVI', '')
        vid_out = os.path.join(out_dir, vid_name)
        if os.path.exists(os.path.join(vid_out, 'meta.pt')):
            n_skip += 1
            continue
        ok = preprocess_video(vp, out_dir, args.data_root, args.frame_size)
        if ok:
            n_ok += 1
        else:
            n_fail += 1

    elapsed = time.time() - t0
    print(f"Done: {n_ok} ok, {n_fail} fail, {n_skip} skip in {elapsed:.0f}s")

    # Copy annotations: map annotations to processed video dirs
    print("Copying annotations...")
    import shutil
    for ann_dir_name in ['Annotation_files', 'Annotations_files', 'annotations', 'Annotations']:
        for ann_dir in glob.glob(os.path.join(args.data_root, '**', ann_dir_name), recursive=True):
            for ann_file in glob.glob(os.path.join(ann_dir, '*.txt')):
                base = os.path.basename(ann_file)
                # Build the scene-prefixed video name the same way
                # Annotation dir is like Le2i/Coffee_room_01/Coffee_room_01/Annotation_files/
                # Video dir is like Le2i/Coffee_room_01/Coffee_room_01/Videos/
                ann_parent = os.path.dirname(ann_dir)  # e.g., Coffee_room_01/Coffee_room_01
                rel_ann = os.path.relpath(ann_parent, args.data_root)
                vid_name = rel_ann.replace('/', '_').replace('\\', '_') + '_' + base.replace('.txt', '')
                vid_out = os.path.join(out_dir, vid_name)
                if os.path.isdir(vid_out):
                    shutil.copy2(ann_file, os.path.join(vid_out, 'annotation.txt'))

    print("Done!")


if __name__ == '__main__':
    main()
