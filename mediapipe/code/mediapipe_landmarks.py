"""
Given a photo, finds a hand with MediaPipe and saves its 21 landmarks
(x, y, z) to a JSON file.

MediaPipe outputs a real x, y, z per landmark:
  x, y -- normalized 0-1 by image width/height
  z    -- relative depth; roughly 0 at the wrist, negative = closer to
           the camera than the wrist, positive = further away

Usage:
    python mediapipe_landmarks.py <input_image> <output_json>

Setup:
    python -m venv venv
    venv\Scripts\activate       (Windows)   or   source venv/bin/activate   (Mac/Linux)
    pip install -r requirements.txt
    python mediapipe_landmarks.py <input_image> <output_json>

The hand-tracking model (../model/hand_landmarker.task) is already included
in this repo -- originally downloaded from Google's own public model zoo:
https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
"""
import sys
import json
from pathlib import Path

import cv2
import mediapipe as mp
from PIL import Image, ImageDraw

MODEL_PATH = Path(__file__).resolve().parent.parent / "model" / "hand_landmarker.task"

# Which landmark connects to which, for drawing the skeleton -- same 20
# bones as the rest of this project (wrist outward, 4 per finger).
HAND_BONES = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
]


def extract_landmarks(image_path):
    """Returns a list of 21 [x, y, z] points, or None if no hand was found."""
    # Configure the landmarker for a single still image with at most one
    # hand -- running_mode=IMAGE (not VIDEO/LIVE_STREAM) is what makes
    # detect() below a plain one-shot call instead of needing a timestamp.
    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_hands=1,
    )
    # MediaPipe expects RGB; OpenCV loads images as BGR, so convert before
    # wrapping the pixels in MediaPipe's own Image type and running detection.
    with mp.tasks.vision.HandLandmarker.create_from_options(options) as landmarker:
        bgr_image = cv2.imread(str(image_path))
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        result = landmarker.detect(mp_image)

    if not result.hand_landmarks:
        return None

    hand = result.hand_landmarks[0]  # first (and only, num_hands=1) detected hand
    return [[point.x, point.y, point.z] for point in hand]


def print_coordinates(landmarks):
    print(f"{'point':>6} {'x':>10} {'y':>10} {'z':>10}")
    for i, (x, y, z) in enumerate(landmarks):
        print(f"{i:>6} {x:10.5f} {y:10.5f} {z:10.5f}")


def draw_skeleton(image_path, landmarks, output_png):
    """Draws the 21 points and the bones between them on top of the photo,
    for a quick visual check that the detection looks right."""
    # Landmarks are stored normalized 0-1; multiply by the photo's own
    # width/height to get real pixel positions to draw at.
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    draw = ImageDraw.Draw(img)
    r = int(min(w, h) * 0.012)

    # Draw a line for every bone in HAND_BONES, connecting the two
    # landmarks it runs between.
    for start, end in HAND_BONES:
        x1, y1, _ = landmarks[start]
        x2, y2, _ = landmarks[end]
        draw.line([(x1 * w, y1 * h), (x2 * w, y2 * h)], fill=(255, 140, 0), width=4)
    # Draw a dot at each of the 21 landmarks on top of the bones.
    for x, y, _ in landmarks:
        px, py = x * w, y * h
        draw.ellipse([px - r, py - r, px + r, py + r], fill=(255, 0, 0), outline=(255, 255, 255))

    img.save(output_png)


# Require exactly the two command-line arguments the script needs; print
# usage and stop otherwise.
if len(sys.argv) != 3:
    print("Usage: python mediapipe_landmarks.py <input_image> <output_json>")
    sys.exit(1)

input_image, output_json = sys.argv[1], sys.argv[2]

# Run the extraction; if no hand was found, say so and stop rather than
# writing bad output files.
landmarks = extract_landmarks(input_image)
if landmarks is None:
    print("No hand detected.")
    sys.exit(1)

print("MediaPipe: hand detected, 21 landmarks\n")
print_coordinates(landmarks)

# save the coordinates for whatever runs next in the pipeline
with open(output_json, "w") as f:
    json.dump({"source": "mediapipe", "landmarks": landmarks}, f, indent=2)
print(f"\nsaved: {output_json}")

# optional visual check, saved next to the JSON with the same name
output_png = str(Path(output_json).with_suffix(".png"))
draw_skeleton(input_image, landmarks, output_png)
print(f"saved: {output_png}")
