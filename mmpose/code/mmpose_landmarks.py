"""
Given a photo, finds the hand with MMPose, prints the 21 keypoint
coordinates, saves them to JSON, and draws the skeleton on the photo.

Usage:
    python mmpose_landmarks.py <input_image> <output_json>
"""
import sys
import json
from pathlib import Path

from mmpose.apis import MMPoseInferencer
from PIL import Image, ImageDraw, ImageFont

# Which landmark connects to which, for drawing the skeleton -- 4 bones per
# finger, each running from the wrist outward.
HAND_BONES = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
]
LABELS = ["Wrist"] + [f"{finger}_{i}" for finger in ["Thumb", "Index", "Middle", "Ring", "Pinky"] for i in range(4)]

# Below this average per-keypoint confidence, treat it as "no hand" rather
# than trust the coordinates -- see the comment on extract() for why.
MIN_AVG_KEYPOINT_SCORE = 0.20


def extract(photo_path):
    """Returns 21 [x, y] keypoints, or None if no hand was actually found.

    MMPoseInferencer's raw output can't be trusted as-is: when its detector
    finds no hand, it silently falls back to treating the ENTIRE IMAGE as
    the hand's bounding box (with a fake, maxed-out confidence) instead of
    reporting "nothing found" -- so the landmark model then dutifully places
    21 points somewhere in whatever picture it's given, hand or not. The two
    checks below catch that: one looks for a bounding box that's suspiciously
    the exact size of the whole photo, the other checks the keypoints' own
    confidence is actually high enough to trust.
    """
    # Open the photo and note its width/height -- needed below to recognize
    # the whole-image-fallback case.
    image = Image.open(photo_path)
    w, h = image.size

    # Run MMPose's hand pipeline (detector + landmark regressor) on the photo
    # and pull out the first (only) prediction. No detection at all -> None.
    inferencer = MMPoseInferencer("hand")  # auto-downloads + caches the detector and landmark model
    result = next(inferencer(str(photo_path)))
    preds = result["predictions"][0]
    if len(preds) == 0:
        return None

    pred = preds[0]

    # Reject the whole-image-fallback case: a bounding box that matches the
    # photo's full width/height almost exactly means the detector actually
    # found nothing and MMPose silently substituted the whole image instead.
    bbox = pred["bbox"][0]
    is_whole_image_fallback = (
        abs(bbox[0]) < 2 and abs(bbox[1]) < 2
        and abs(bbox[2] - w) < 2 and abs(bbox[3] - h) < 2
    )
    if is_whole_image_fallback:
        return None

    # Reject low-confidence keypoints too -- a real detection whose average
    # per-point confidence is still too low to trust.
    avg_keypoint_score = sum(pred["keypoint_scores"]) / len(pred["keypoint_scores"])
    if avg_keypoint_score < MIN_AVG_KEYPOINT_SCORE:
        return None

    return pred["keypoints"]  # 21 x [x, y], raw pixel coordinates


def print_coordinates(keypoints):
    print(f"{'point':10s} {'x':>10s} {'y':>10s}")
    for label, (x, y) in zip(LABELS, keypoints):
        print(f"{label:10s} {x:10.1f} {y:10.1f}")


def draw_skeleton(photo_path, keypoints, output_path):
    """Draws the 21 points and the bones between them on top of the photo,
    for a quick visual check that the detection looks right."""
    # Open the photo and size the dots/font relative to it, so the overlay
    # looks right whether the photo is a phone-camera shot or a tiny crop.
    img = Image.open(photo_path).convert("RGB")
    w, h = img.size
    draw = ImageDraw.Draw(img)
    radius = int(min(w, h) * 0.012)

    # Prefer a real bold font for the point numbers; fall back to Pillow's
    # built-in default if that exact font file isn't on this machine.
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", int(min(w, h) * 0.018))
    except OSError:
        font = ImageFont.load_default()

    # Draw a line for every bone in HAND_BONES, connecting the two keypoints
    # it runs between.
    for start, end in HAND_BONES:
        x1, y1 = keypoints[start]
        x2, y2 = keypoints[end]
        draw.line([(x1, y1), (x2, y2)], fill=(255, 210, 0), width=6)

    # Draw a numbered dot at each of the 21 keypoints on top of the bones.
    for i, (x, y) in enumerate(keypoints):
        draw.ellipse([x - radius, y - radius, x + radius, y + radius],
                     fill=(0, 120, 255), outline=(255, 255, 255), width=4)
        draw.text((x + radius + 4, y - radius), str(i), font=font, fill=(255, 255, 0))

    img.save(output_path)


# Require exactly the two command-line arguments the script needs; print
# usage and stop otherwise.
if len(sys.argv) != 3:
    print("Usage: python mmpose_landmarks.py <input_image> <output_json>")
    sys.exit(1)

photo_path = Path(sys.argv[1])
output_json = Path(sys.argv[2])

# Run the extraction; if no hand was found (or found but not trustworthy),
# say so and stop rather than writing bad output files.
keypoints = extract(photo_path)
if keypoints is None:
    print("MMPose: no hand detected in this photo.")
    sys.exit(1)

print("MMPose: hand detected, 21 keypoints\n")
print_coordinates(keypoints)

# save the coordinates for whatever runs next in the pipeline
with open(output_json, "w") as f:
    json.dump({"source": "mmpose", "landmarks": keypoints}, f, indent=2)
print(f"\nsaved: {output_json}")

# optional visual check, saved next to the JSON with the same name
output_png = output_json.with_suffix(".png")
draw_skeleton(photo_path, keypoints, output_png)
print(f"saved: {output_png}")
