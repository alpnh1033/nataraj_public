"""
Rubric-weighted scoring: compares an attempt's bone vectors against a
mudra's reference bone vectors using cosine similarity, weighted by a
teacher-authored rubric, to produce both a final grade and a per-bone
breakdown feedback can be built from.
"""
import base64
import json
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from mudra_core import HAND_BONES, bone_angle_deltas, directional_feedback

CLASS_REFERENCE_VECTORS_PATH = Path(__file__).resolve().parent / "class_reference_vectors.json"
_class_reference_vectors = None


def _get_class_reference_vectors():
    """
    Lazily loads the per-class mean bone vector -- one 20x2 array per mudra
    class, computed from the training data (average unit bone vector across
    all training examples of that class). This is "the reference" a
    submitted attempt is scored and given feedback against.
    """
    global _class_reference_vectors
    if _class_reference_vectors is None:
        with open(CLASS_REFERENCE_VECTORS_PATH) as f:
            raw = json.load(f)
        _class_reference_vectors = {name: np.array(vecs) for name, vecs in raw.items()}
    return _class_reference_vectors


def directional_feedback_for_class(target_vectors, target_mudra, weak_bone_names):
    """
    Same idea as mudra_core.directional_feedback(), but against a per-class
    average reference vector instead of one reference photo. Returns None
    for a class with no computed reference, rather than raising.
    """
    refs = _get_class_reference_vectors()
    if target_mudra not in refs:
        return None
    deltas = bone_angle_deltas(refs[target_mudra], target_vectors)
    finger_names = list(dict.fromkeys(bone.split("_")[0] for bone in weak_bone_names))
    return [directional_feedback(deltas, finger) for finger in finger_names]

# One rubric weight per bone, same order as HAND_BONES.
BONE_NAMES = [
    f"{finger}_{i}"
    for finger, count in [("Thumb", 4), ("Index", 4), ("Middle", 4), ("Ring", 4), ("Pinky", 4)]
    for i in range(count)
]
assert len(BONE_NAMES) == len(HAND_BONES) == 20


def default_rubric():
    """Uniform weighting — every bone matters equally. Used when no
    teacher-authored rubric exists yet for a given mudra."""
    return {name: 1.0 for name in BONE_NAMES}


def save_rubric(s3_client, bucket, mudra_name, weights):
    """Writes a teacher-authored rubric to S3 (rubrics/mudra/<name>.json),
    overwriting any existing rubric for that mudra."""
    key = f"rubrics/mudra/{mudra_name}.json"
    s3_client.put_object(
        Bucket=bucket,
        Key=key,
        Body=json.dumps(weights).encode(),
        ContentType="application/json",
    )


def load_rubric(s3_client, bucket, mudra_name):
    """Loads a teacher-authored rubric JSON from S3 (rubrics/mudra/<name>.json).
    Falls back to a uniform default if none has been authored yet."""
    key = f"rubrics/mudra/{mudra_name}.json"
    try:
        obj = s3_client.get_object(Bucket=bucket, Key=key)
        weights = json.loads(obj["Body"].read())
        merged = default_rubric()
        merged.update(weights)
        return merged
    except s3_client.exceptions.NoSuchKey:
        return default_rubric()


def compute_weighted_score(attempt_vectors, target_mudra, rubric_weights):
    """
    Rubric-weighted mean of per-bone geometric correctness against
    target_mudra's reference vector:

        c_i = (1 + cos(a_i, r_i)) / 2                 # per-bone correctness, bone i, in [0,1]
        G   = 100 * (sum_i w_i * c_i) / (sum_i w_i)   # final score

    `attempt_vectors` is the attempt's unit bone vectors
    (mudra_core.bone_vectors()). The reference r_i is target_mudra's
    per-class mean bone vector (class_reference_vectors.json) -- the same
    reference directional_feedback_for_class() uses, so the score and the
    directional feedback agree with each other.

    Returns (final_score_0_100, per_bone_breakdown), breakdown sorted worst
    (highest loss) first so callers can take the top N to flag.
    """
    reference_vectors = _get_class_reference_vectors()[target_mudra]
    total_weight = sum(rubric_weights.get(name, 1.0) for name in BONE_NAMES)

    per_bone = {}
    for name, ref_vec, att_vec in zip(BONE_NAMES, reference_vectors, attempt_vectors):
        cos_sim = np.dot(ref_vec, att_vec) / (np.linalg.norm(ref_vec) * np.linalg.norm(att_vec))
        cos_sim = np.clip(cos_sim, -1.0, 1.0)  # guards against float rounding pushing slightly past +-1
        correctness = (1.0 + cos_sim) / 2.0
        weight = rubric_weights.get(name, 1.0)
        per_bone[name] = {
            "bone": name,
            "correctness": float(correctness),
            "loss": float(weight * (1.0 - correctness) / total_weight),
            "rubric_weight": weight,
        }

    final_score = 100.0 * sum(
        per_bone[name]["rubric_weight"] * per_bone[name]["correctness"] for name in BONE_NAMES
    ) / total_weight

    breakdown = sorted(per_bone.values(), key=lambda b: b["loss"], reverse=True)
    return round(float(final_score), 2), breakdown


_BONE_NAME_TO_INDEX = {name: i for i, name in enumerate(BONE_NAMES)}
_MAX_ANNOTATED_DIMENSION = 800  # keep the response small — this is a phone-viewed annotation, not a print


def _load_marker_font(radius):
    try:
        return ImageFont.load_default(size=max(20, radius * 2))
    except TypeError:
        return ImageFont.load_default()


def annotate_weak_joints(image_bgr, landmarks, markers):
    """
    `markers` is a list of (number, bone_name). Each marked bone's two joints
    get a red circle, and the number is written beside its end joint so it
    matches the numbered feedback bullet for that finger. `landmarks` is the
    raw Nx2 array from mudra_core.extract_landmarks (normalized 0-1 coordinates).
    Returns a PIL Image (RGB), resized to a max dimension to keep the response
    payload small.
    """
    h, w = image_bgr.shape[:2]
    image_rgb = image_bgr[:, :, ::-1]  # BGR -> RGB, no cv2 needed for a channel flip
    pil_image = Image.fromarray(image_rgb)
    draw = ImageDraw.Draw(pil_image)

    radius = max(6, int(min(w, h) * 0.02))
    font = _load_marker_font(radius)
    pad = max(4, radius // 3)
    for number, bone_name in markers:
        bone_idx = _BONE_NAME_TO_INDEX.get(bone_name)
        if bone_idx is None:
            continue
        start_idx, end_idx = HAND_BONES[bone_idx]
        sx, sy = int(landmarks[start_idx][0] * w), int(landmarks[start_idx][1] * h)
        ex, ey = int(landmarks[end_idx][0] * w), int(landmarks[end_idx][1] * h)
        draw.line([(sx, sy), (ex, ey)], fill=(255, 0, 0), width=max(3, radius // 3))
        draw.ellipse([ex - radius, ey - radius, ex + radius, ey + radius], outline=(255, 0, 0), width=4)

        text = str(number)
        left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
        tw, th = right - left, bottom - top
        tx = min(max(ex + radius + 8, pad), w - tw - 2 * pad - 1)
        ty = min(max(ey - th // 2 - pad, pad), h - th - 3 * pad - 1)
        draw.rectangle([tx - pad, ty - pad, tx + tw + pad, ty + th + pad], fill=(0, 0, 0))
        draw.text((tx - left, ty - top), text, fill=(255, 255, 0), font=font)

    if max(pil_image.size) > _MAX_ANNOTATED_DIMENSION:
        pil_image.thumbnail((_MAX_ANNOTATED_DIMENSION, _MAX_ANNOTATED_DIMENSION))

    return pil_image


def image_to_base64_jpeg(pil_image, quality=80):
    """JPEG, not PNG — photographic content compresses far smaller as JPEG,
    which matters since this travels through the grading API's response and
    ultimately a phone's data connection."""
    buf = BytesIO()
    pil_image.convert("RGB").save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode()
