"""
Nataraj Project - Shared Mudra Scoring Engine
Extracts hand landmarks from a photo via MMPose, converts them into
scale-invariant bone vectors, and scores how closely they match a
reference mudra.
"""

from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

# MMPose needs two model checkpoints (a hand detector + a landmark
# regressor). The real deployed Lambda bundles both locally; this public
# copy doesn't (keeps the repo small), so _get_mmpose_inferencer() below
# auto-downloads them instead when they're missing.
_MMPOSE_MODELS_DIR = Path(__file__).resolve().parent / "models" / "mmpose"
_MMPOSE_POSE_WEIGHTS = _MMPOSE_MODELS_DIR / "rtmpose-m_simcc-hand5_pt-aic-coco_210e-256x256-74fb594_20230320.pth"
_MMPOSE_DET_WEIGHTS = _MMPOSE_MODELS_DIR / "rtmdet_nano_8xb32-300e_hand-267f9c8f.pth"

# Below this average per-keypoint confidence, treat it as "no hand" rather
# than trust the coordinates -- needed alongside the whole-image-fallback
# check below, since a confidence threshold alone doesn't catch that case
# (its confidence is fabricated at a fixed 1.0 regardless of image content).
MIN_AVG_KEYPOINT_SCORE = 0.20

_mmpose_inferencer = None
_MMPoseInferencer = None


def _ensure_mmpose_imported():
    # Deferred because torch/mmcv/mmpose are heavy imports -- only pay that
    # cost if this function actually gets called.
    global _MMPoseInferencer
    if _MMPoseInferencer is None:
        from mmpose.apis import MMPoseInferencer as _Inferencer
        _MMPoseInferencer = _Inferencer


def _get_mmpose_inferencer():
    global _mmpose_inferencer
    _ensure_mmpose_imported()
    if _mmpose_inferencer is None:
        # pose2d="hand" resolves MMPose's own bundled hand-pose config;
        # pose2d_weights points it at our local checkpoint instead. The
        # detector needs det_model and det_cat_ids supplied explicitly too
        # (not just det_weights) -- MMPoseInferencer silently falls back to
        # its own defaults otherwise, or rejects every real detection.
        if _MMPOSE_POSE_WEIGHTS.exists() and _MMPOSE_DET_WEIGHTS.exists():
            from mmpose.apis.inferencers.utils.default_det_models import default_det_models
            det_info = default_det_models["hand"]

            _mmpose_inferencer = _MMPoseInferencer(
                pose2d="hand",
                pose2d_weights=str(_MMPOSE_POSE_WEIGHTS),
                det_model=det_info["model"],
                det_weights=str(_MMPOSE_DET_WEIGHTS),
                det_cat_ids=det_info["cat_ids"],
            )
        else:
            # This public copy doesn't bundle the checkpoint files -- let
            # MMPoseInferencer auto-download and cache them instead (same
            # fallback nataraj_public/mmpose/mmpose_landmarks.py uses).
            _mmpose_inferencer = _MMPoseInferencer("hand")
    return _mmpose_inferencer

# Bone connections between joints, grouped along each finger from the wrist outward.
HAND_BONES = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # Index
    (0, 9), (9, 10), (10, 11), (11, 12),   # Middle
    (0, 13), (13, 14), (14, 15), (15, 16), # Ring
    (0, 17), (17, 18), (18, 19), (19, 20), # Pinky
]

# Index ranges into HAND_BONES/similarities for each finger — 4 bones per finger, in order.
FINGER_BONE_RANGES = {
    "Thumb": (0, 4),
    "Index": (4, 8),
    "Middle": (8, 12),
    "Ring": (12, 16),
    "Pinky": (16, 20),
}

# Fingers scoring below this are candidates for being called out as needing work.
FOCUS_THRESHOLD = 80.0

# A finger must also score at least this many points below the mean of all five
# fingers to be singled out — otherwise a uniformly low score (e.g. the whole
# hand's rotation/angle differs from the reference) would flag every finger,
# which isn't actionable feedback for the student.
OUTLIER_MARGIN = 4.0


class HandNotDetectedError(Exception):
    """Raised when MMPose finds no hand in a given image."""


def _pil_to_bgr(pil_image):
    # cv2.imread/imdecode ignore EXIF orientation, which phone camera photos
    # commonly rely on (pixels stored "sideways" with a rotation tag rather
    # than physically rotated) — going through PIL for every format, not just
    # HEIC, and applying exif_transpose fixes hand detection on phone photos
    # that would otherwise be handed to the detector rotated 90/180/270 degrees.
    pil_image = ImageOps.exif_transpose(pil_image)
    return cv2.cvtColor(np.array(pil_image.convert("RGB")), cv2.COLOR_RGB2BGR)


def load_image_from_path(path):
    """Loads an image from disk as a BGR numpy array, EXIF-orientation-corrected.
    Supports HEIC/HEIF (iPhone photos) via Pillow."""
    image = Image.open(path)
    return _pil_to_bgr(image)


def load_image_from_bytes(data, filename=""):
    """Decodes an in-memory image (e.g. an uploaded file) as a BGR numpy array,
    EXIF-orientation-corrected."""
    image = Image.open(BytesIO(data))
    return _pil_to_bgr(image)


def extract_landmarks(image):
    """
    Extracts the 21 2D hand landmarks from a BGR image via MMPose, as
    (x, y) coordinates normalized to 0-1 by image width/height.

    No z coordinate: MMPose doesn't estimate hand depth (MediaPipe does,
    but this project uses MMPose). Raises HandNotDetectedError if no hand
    is found, including MMPose's "whole image as fallback bounding box"
    behavior when its detector actually finds nothing -- that case reports
    a fake 1.0 confidence, so it has to be checked for separately from the
    real per-keypoint confidence check below it.
    """
    h, w = image.shape[:2]
    result = next(_get_mmpose_inferencer()(image))
    preds = result["predictions"][0]
    if not preds:
        raise HandNotDetectedError("No hand detected in image.")

    pred = preds[0]
    bbox = pred["bbox"][0]
    is_whole_image_fallback = (
        abs(bbox[0]) < 2 and abs(bbox[1]) < 2
        and abs(bbox[2] - w) < 2 and abs(bbox[3] - h) < 2
    )
    if is_whole_image_fallback:
        raise HandNotDetectedError("No hand detected in image.")

    avg_keypoint_score = sum(pred["keypoint_scores"]) / len(pred["keypoint_scores"])
    if avg_keypoint_score < MIN_AVG_KEYPOINT_SCORE:
        raise HandNotDetectedError("No hand detected in image.")

    keypoints = np.array(pred["keypoints"])  # 21 x [x, y], raw pixels
    return keypoints / np.array([w, h])  # normalize to 0-1, matching MediaPipe's convention


def bone_vectors(landmarks):
    """Converts absolute joint positions into scale-invariant unit direction vectors."""
    vectors = []
    for start, end in HAND_BONES:
        vec = landmarks[end] - landmarks[start]
        norm = np.linalg.norm(vec)
        vectors.append(vec / norm if norm > 0 else vec)
    return np.array(vectors)


def _cos_sim_to_pct(cos_sim):
    """Remaps the [-1, 1] cosine similarity range to a [0, 100] percentage match."""
    return (cos_sim + 1) / 2 * 100


def score_details(reference_vectors, target_vectors):
    """
    Computes cosine similarity across all corresponding finger-joint vectors,
    returning both the overall score and a per-finger breakdown so feedback
    can point at exactly which finger to fix, not just an aggregate number.
    """
    similarities = np.array([
        np.dot(ref_vec, tgt_vec) / (np.linalg.norm(ref_vec) * np.linalg.norm(tgt_vec))
        for ref_vec, tgt_vec in zip(reference_vectors, target_vectors)
    ])

    overall = _cos_sim_to_pct(np.mean(similarities))
    fingers = {
        name: _cos_sim_to_pct(np.mean(similarities[start:end]))
        for name, (start, end) in FINGER_BONE_RANGES.items()
    }
    return overall, fingers


def weak_fingers(fingers, threshold=FOCUS_THRESHOLD, margin=OUTLIER_MARGIN):
    """
    Returns (finger_name, score) pairs that are both below the absolute threshold
    and a genuine outlier relative to the other fingers, worst first. A finger
    that's merely as low as the rest of an evenly-matched hand isn't flagged —
    that points at an overall posture/angle issue instead (see feedback_message).
    """
    mean = np.mean(list(fingers.values()))
    return sorted(
        (
            (name, pct) for name, pct in fingers.items()
            if pct < threshold and pct <= mean - margin
        ),
        key=lambda item: item[1],
    )


def bone_angle_deltas(reference_vectors, target_vectors):
    """
    Per-bone angle (in degrees) between the reference and target hand's
    corresponding bone vectors. 0 degrees = perfectly aligned with the
    reference; 180 = pointing exactly opposite. Turns "this finger scored
    low" into "this finger points N degrees away from the reference."
    """
    angles = []
    for ref_vec, tgt_vec in zip(reference_vectors, target_vectors):
        cos_sim = np.dot(ref_vec, tgt_vec) / (np.linalg.norm(ref_vec) * np.linalg.norm(tgt_vec))
        cos_sim = np.clip(cos_sim, -1.0, 1.0)  # guards against float rounding pushing slightly past +-1
        angles.append(np.degrees(np.arccos(cos_sim)))
    return np.array(angles)


def finger_angle_deltas(bone_deltas):
    """Per-finger mean angle delta (degrees), same finger grouping as score_details()'s `fingers` dict."""
    return {
        name: float(np.mean(bone_deltas[start:end]))
        for name, (start, end) in FINGER_BONE_RANGES.items()
    }


# Human-readable label for each finger's 4 bones, wrist-outward -- matches
# HAND_BONES' per-finger ordering (e.g. Index: (0,5),(5,6),(6,7),(7,8)).
BONE_SEGMENT_LABELS = ["base knuckle", "middle knuckle", "upper knuckle", "fingertip"]


def directional_feedback(bone_deltas, finger_name):
    """
    For one flagged weak finger, names the single most-off bone segment
    within it and its angle -- turns "Thumb is wrong" into "Thumb is 23
    degrees off from the reference, most noticeably at the fingertip."
    """
    start, end = FINGER_BONE_RANGES[finger_name]
    segment_deltas = bone_deltas[start:end]
    worst_local_idx = int(np.argmax(segment_deltas))
    worst_angle = float(segment_deltas[worst_local_idx])
    worst_label = BONE_SEGMENT_LABELS[worst_local_idx]
    mean_angle = float(np.mean(segment_deltas))
    return {
        "finger": finger_name,
        "mean_angle_degrees": round(mean_angle, 1),
        "worst_segment": worst_label,
        "worst_segment_angle_degrees": round(worst_angle, 1),
        "message": (
            f"{finger_name} is {mean_angle:.0f}° off from the reference, "
            f"most noticeably at the {worst_label} ({worst_angle:.0f}°)."
        ),
    }


def feedback_message(score):
    if score >= 90:
        return "Very close to the reference posture."
    elif score >= 75:
        return "Good attempt. Minor adjustments needed to refine the gesture."
    elif score >= 50:
        return "Partial match. Review the reference posture and try again."
    else:
        return "Low match. The gesture differs significantly from the reference."
