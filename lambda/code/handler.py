"""
Lambda for Mudra grading

Rubric weights live in S3 and are read
here at grading time. No generative AI is used anywhere in this file --
grading and feedback are both deterministic.
"""
import base64
import json
import os

import boto3
import joblib

import mudra_core
import rubric_engine

MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "model_artifacts", "mudra", "logistic_regression_model.joblib"
)
S3_BUCKET = os.environ.get("MODELS_BUCKET")  # set this to your own bucket name

_model = None


def _get_model():
    global _model
    if _model is None:
        _model = joblib.load(MODEL_PATH)
    return _model


def _grade_mudra_impl(image_bytes, target_mudra, s3_client):
    """Returns (result_dict, annotated_image_base64). The annotated image is
    kept separate from the result dict so the JSON result stays text-only."""
    image = mudra_core.load_image_from_bytes(image_bytes)
    landmarks = mudra_core.extract_landmarks(image)  # raises HandNotDetectedError if none found
    vectors = mudra_core.bone_vectors(landmarks)
    features = vectors.flatten().reshape(1, -1)

    model = _get_model()
    classes = list(model.classes_)
    if target_mudra not in classes:
        raise ValueError(
            f"Unknown mudra class '{target_mudra}'. Known classes: {classes}"
        )
    class_idx = classes.index(target_mudra)
    proba = model.predict_proba(features)[0][class_idx]

    rubric_weights = rubric_engine.load_rubric(s3_client, S3_BUCKET, target_mudra)
    final_score, breakdown = rubric_engine.compute_weighted_score(
        vectors, target_mudra, rubric_weights
    )
    weakest_bones = breakdown[:5]
    fingers = list(dict.fromkeys(b["bone"].split("_")[0] for b in weakest_bones))

    # Real angular deltas against this class's per-training-data reference
    # vector -- turns "this bone scored low" into "this finger is N degrees
    # off." Soft-fails (None) rather than raising if a class somehow has no
    # computed reference, so grading itself never breaks because of this.
    directional_all = rubric_engine.directional_feedback_for_class(
        vectors, target_mudra, [b["bone"] for b in weakest_bones]
    )

    # Only surface a finger if it's off by enough to actually be correctable --
    # below this, the difference is closer to normal photo-to-photo variation
    # than something a dancer could reliably act on. Without this, a 96%
    # attempt could still get 4-5 "corrections," which reads as nitpicking
    # rather than the near-perfect score it actually reflects.
    directional = [d for d in (directional_all or []) if d["mean_angle_degrees"] >= MIN_FEEDBACK_DEGREES]
    flagged_fingers = {d["finger"] for d in directional}
    ordered_flagged_fingers = [f for f in fingers if f in flagged_fingers]
    markers = [
        (number, next(b["bone"] for b in weakest_bones if b["bone"].startswith(finger + "_")))
        for number, finger in enumerate(ordered_flagged_fingers, start=1)
    ]

    annotated = rubric_engine.annotate_weak_joints(image, landmarks, markers)
    annotated_b64 = rubric_engine.image_to_base64_jpeg(annotated)

    result = {
        "target_mudra": target_mudra,
        "raw_model_probability_pct": round(float(proba) * 100, 2),
        "rubric_weighted_score": final_score,
        "weakest_bones": weakest_bones,
        "directional_feedback": directional,
    }
    return result, annotated_b64


def _response(status_code, body_dict):
    return {
        "statusCode": status_code,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(body_dict),
    }


MIN_MATCH_PROBABILITY_PCT = 50.0
MIN_FEEDBACK_DEGREES = 15.0


def _template_feedback(result, mode):
    mudra = result["target_mudra"]
    score = result["rubric_weighted_score"]
    has_flags = bool(result["directional_feedback"])
    lines = []
    if result["raw_model_probability_pct"] < MIN_MATCH_PROBABILITY_PCT:
        lines.append(
            f"This photo doesn't look like a {mudra} mudra "
            f"(classifier match {result['raw_model_probability_pct']:.0f}%). "
            f"Compare your hand shape with the reference."
        )
    else:
        closing = "Check the joints below for small refinements." if has_flags else "No corrections needed — great form!"
        if mode == "learning":
            lines.append(f"Practice attempt at {mudra}. {mudra_core.feedback_message(score)} {closing}")
        else:
            lines.append(f"{mudra}: score {score:.0f}/100. {mudra_core.feedback_message(score)} {closing}")
    for number, item in enumerate(result["directional_feedback"] or [], start=1):
        line = (
            f"{number}. {item['finger']}: {item['mean_angle_degrees']:.0f}° off overall, "
            f"most at the {item['worst_segment']} ({item['worst_segment_angle_degrees']:.0f}°)."
        )
        if mode == "learning":
            line += (
                f" Try adjusting the {item['finger'].lower()} at the {item['worst_segment']} "
                f"until it matches the reference."
            )
        lines.append(line)
    return "\n".join(lines)


def _grade_route(event, payload, s3_client):
    """
    Supports two request shapes:
    1. JSON body {"image_base64": ..., "target_mudra": ...} — used by test
       scripts and any client that can build JSON + base64 easily.
    2. Raw image bytes as the POST body + ?target_mudra=... query param —
       used by the MIT App Inventor phone app, whose Web component
       (Web.PostFile) sends a file's raw bytes directly with no built-in
       JSON/base64 encoding available.
    """
    query_params = event.get("queryStringParameters") or {}
    if "target_mudra" in query_params:
        target_mudra = query_params["target_mudra"]
        raw_body = event.get("body", "")
        image_bytes = base64.b64decode(raw_body) if event.get("isBase64Encoded") else raw_body.encode("latin-1")
    else:
        try:
            image_bytes = base64.b64decode(payload["image_base64"])
            target_mudra = payload["target_mudra"]
        except KeyError as e:
            return _response(400, {"error": f"Missing required field: {e}"})

    mode = query_params.get("mode", "evaluation")
    try:
        result, annotated_b64 = _grade_mudra_impl(image_bytes, target_mudra, s3_client)
    except mudra_core.HandNotDetectedError:
        return _response(422, {"error": "No hand detected in the image. Please retake the photo."})
    except ValueError as e:
        return _response(400, {"error": str(e)})

    feedback = _template_feedback(result, mode)
    print(json.dumps({
        "evaluation": {
            "mode": mode,
            "target_mudra": target_mudra,
            "rubric_weighted_score": result["rubric_weighted_score"],
            "raw_model_probability_pct": result["raw_model_probability_pct"],
            "feedback": feedback,
        }
    }))
    return _response(200, {
        "result": result,
        "feedback": feedback,
        "annotated_image_base64": annotated_b64,
    })


def handler(event, context):
    query_params = event.get("queryStringParameters") or {}

    # Cheap shared-secret check, done first and before any real work (MMPose
    # calls cost money) — a Lambda Function URL has no built-in API-key or
    # usage-plan concept at all (that's an API Gateway feature), so this is
    # the equivalent for a bare Function URL: a query param checked against
    # a Lambda environment variable, rotatable without rebuilding the
    # container image.
    expected_key = os.environ.get("API_KEY")
    if expected_key and query_params.get("api_key") != expected_key:
        return _response(401, {"error": "Invalid or missing API key."})

    # Raw-image-body mode (App Inventor's Web.PostFile) is signaled by
    # target_mudra arriving as a query param rather than JSON — skip JSON
    # parsing entirely in that case, since the body is binary image data.
    if "target_mudra" in query_params:
        payload = {}
    else:
        body = event.get("body", "{}")
        if event.get("isBase64Encoded"):
            body = base64.b64decode(body).decode("utf-8")
        payload = json.loads(body)

    s3_client = boto3.Session().client("s3")
    return _grade_route(event, payload, s3_client)
