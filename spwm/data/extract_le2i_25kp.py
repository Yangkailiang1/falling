"""
Re-extract Le2i keypoints as 25 NTU-aligned joints using MediaPipe Tasks API.

Output: (T, 25, 2) keypoints + (T, 25) confidence in le2i_keypoints_25/
Matches NTU120 format: both have 25 joints x 3 channels per joint.

Usage:
  python3 -m spwm.data.extract_le2i_25kp
"""

import os
import re
import sys
import json
import time
import numpy as np
from pathlib import Path

import av
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions, RunningMode
from mediapipe import Image, ImageFormat


def split_name_to_avi(vname: str) -> str:
    """Map le2i_split.json video name to actual AVI path.
    'Coffee_room_01_Coffee_room_01_Videos_video (1)' -> 'Le2i/Coffee_room_01/Coffee_room_01/Videos/video (1).avi'
    """
    m = re.match(r'(.+?)_\1_Videos_video \((\d+)\)', vname)
    if m:
        scene, num = m.group(1), m.group(2)
        return f"Le2i/{scene}/{scene}/Videos/video ({num}).avi"
    m = re.match(r'(.+?)_(.+?)_Videos_video \((\d+)\)', vname)
    if m:
        scene1, scene2, num = m.group(1), m.group(2), m.group(3)
        return f"Le2i/{scene1}/{scene2}/Videos/video ({num}).avi"
    return f"Le2i/{vname}.avi"


# MediaPipe 33-landmark -> NTU 25-joint mapping
# NTU order: base_spine(0), mid_spine(1), neck(2), head(3),
#   L_shoulder(4), L_elbow(5), L_wrist(6), L_hand(7),
#   R_shoulder(8), R_elbow(9), R_wrist(10), R_hand(11),
#   L_hip(12), L_knee(13), L_ankle(14), L_foot(15),
#   R_hip(16), R_knee(17), R_ankle(18), R_foot(19),
#   spine_shoulder(20), L_hand_tip(21), L_thumb(22), R_hand_tip(23), R_thumb(24)

def mp33_to_ntu25(lm33: np.ndarray) -> np.ndarray:
    """(33, 4) MediaPipe -> (25, 3) NTU, where channel 2 = z*depth (as confidence proxy)."""

    def pt(i):
        return lm33[i, :3].copy()

    def mid(i, j):
        return (pt(i) + pt(j)) / 2

    kp = np.zeros((25, 3), dtype=np.float32)

    hip_mid = mid(23, 24)
    shoulder_mid = mid(11, 12)

    kp[0] = hip_mid
    kp[1] = hip_mid * 0.5 + shoulder_mid * 0.5
    kp[2] = shoulder_mid
    kp[3] = pt(0)                     # nose as head
    kp[20] = shoulder_mid * 0.25 + hip_mid * 0.75

    kp[4] = pt(11); kp[5] = pt(13); kp[6] = pt(15)
    kp[7] = mid(15, 17)
    kp[21] = pt(19); kp[22] = pt(21)

    kp[8] = pt(12); kp[9] = pt(14); kp[10] = pt(16)
    kp[11] = mid(16, 18)
    kp[23] = pt(20); kp[24] = pt(22)

    kp[12] = pt(23); kp[13] = pt(25); kp[14] = pt(27)
    kp[15] = mid(27, 31)

    kp[16] = pt(24); kp[17] = pt(26); kp[18] = pt(28)
    kp[19] = mid(28, 32)

    return kp


def extract_video(video_path: str, model_path: str = "pose_landmarker_lite.task"):
    """Extract 25-joint NTU-aligned keypoints from a video."""
    container = av.open(video_path)
    stream = container.streams.video[0]
    fps = float(stream.average_rate)

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=RunningMode.VIDEO,
        num_poses=1,
    )
    detector = PoseLandmarker.create_from_options(options)

    all_kp, all_conf = [], []
    frame_idx = 0

    for frame in container.decode(stream):
        img = frame.to_ndarray(format="rgb24")
        h, w = img.shape[:2]

        mp_img = Image(image_format=ImageFormat.SRGB, data=img)
        ts_ms = int(frame_idx * 1000 / fps)
        result = detector.detect_for_video(mp_img, ts_ms)

        if result.pose_landmarks and len(result.pose_landmarks) > 0:
            lm33 = np.zeros((33, 4), dtype=np.float32)
            for i, lm in enumerate(result.pose_landmarks[0]):
                lm33[i] = [lm.x * w, lm.y * h, lm.z, lm.visibility or 0.0]
            kp25 = mp33_to_ntu25(lm33)
        else:
            kp25 = np.zeros((25, 3), dtype=np.float32)

        all_kp.append(kp25[:, :2])
        all_conf.append(kp25[:, 2])
        frame_idx += 1

    detector.close()
    container.close()
    return np.array(all_kp, dtype=np.float32), np.array(all_conf, dtype=np.float32)


def main():
    le2i_root = Path("Le2i")
    output_root = Path("le2i_keypoints_25")
    split_json = "le2i_split.json"
    model_path = "pose_landmarker_lite.task"

    if not os.path.exists(model_path):
        print(f"ERROR: {model_path} not found. Download from:")
        print("  wget https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task")
        return

    with open(split_json) as f:
        split_data = json.load(f)

    kp_scenes = {"Coffee_room_01", "Coffee_room_02", "Home_01", "Home_02"}
    processed = 0
    t0 = time.time()

    for vname, info in split_data["videos"].items():
        if not any(vname.startswith(s) for s in kp_scenes):
            continue
        avi_path = split_name_to_avi(vname)
        if not os.path.exists(avi_path):
            continue

        label = "fall" if info.get("is_fall", False) else "normal"
        processed += 1
        elapsed = time.time() - t0
        print(f"[{processed}] {vname} ({label}) [total: {elapsed:.0f}s]")

        try:
            kp_arr, conf_arr = extract_video(str(avi_path), model_path)
        except Exception as e:
            print(f"  ERROR: {e}")
            continue

        os.makedirs(output_root / label, exist_ok=True)
        vid_name = vname.replace(" ", "_").replace("(", "").replace(")", "")
        np.save(output_root / label / f"{vid_name}_keypoints.npy", kp_arr)
        np.save(output_root / label / f"{vid_name}_confs.npy", conf_arr)
        print(f"  -> {kp_arr.shape}, {output_root / label / vid_name}")

    print(f"\nDone. {processed} videos in {time.time()-t0:.0f}s -> {output_root}/")


if __name__ == "__main__":
    main()
