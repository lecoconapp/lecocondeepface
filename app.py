# DeepFace gender-analysis server for Le Cocon's user verification.
#
# Endpoint:  POST /analyze
#   Accepts:  multipart/form-data with `image` (file), OR JSON {"image_base64": "..."}
#   Returns:  {"ok": true, "gender": "Woman"|"Man"|null, "confidence": <0-100>|null,
#              "faceFound": bool, "note": "...", "error": null}
#             or {"ok": false, "error": "..."}
#
# Run locally (downloads models on first request):
#   pip install flask deepface opencv-python-headless
#   python app.py
#
# The Supabase edge function calls this at the DEEPFACE_URL secret.

import base64
import io
import os

from flask import Flask, jsonify, request

app = Flask(__name__)

# Import lazily so that if deepface isn't installed yet, the server still
# starts and returns a clear error message instead of crashing.
DEEP_FACE = None


def _get_deepface():
    global DEEP_FACE
    if DEEP_FACE is None:
        try:
            from deepface import DeepFace
            DEEP_FACE = DeepFace
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(f"Could not load DeepFace: {e}") from e
    return DEEP_FACE


def _load_image_bytes():
    """Return raw image bytes from either multipart `image` or `image_base64`."""
    if request.files and "image" in request.files:
        return request.files["image"].read()
    payload = request.get_json(silent=True) or {}
    b64 = payload.get("image_base64")
    if b64:
        return base64.b64decode(b64)
    return None


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"ok": True, "service": "deepface-verify"})


@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        raw = _load_image_bytes()
        if not raw:
            return jsonify(
                {"ok": False, "error": "No image provided (multipart 'image' or JSON 'image_base64')."}
            ), 400

        df = _get_deepface()
        image_path = _bytes_to_temp(raw)

        # Analyze only the gender attribute. enforce_detection=False means a
        # missing face won't raise; it returns an empty list instead.
        try:
            results = df.analyze(
                img_path=image_path,
                actions=["gender"],
                enforce_detection=False,
                silent=True,
            )
        except Exception as e:  # noqa: BLE001 - some deepface versions raise on no face
            results = []

        # DeepFace pre-2023 returns a dict; newer versions return a list.
        if isinstance(results, list):
            results = results[0] if results else None
        if not results:
            return jsonify(
                {
                    "ok": True,
                    "gender": None,
                    "confidence": None,
                    "faceFound": False,
                    "note": "No clear face detected.",
                }
            )

        gender_obj = results.get("gender") or {}
        # DeepFace returns {Woman: prob, Man: prob} (probs sum to ~1).
        prob_woman = float(gender_obj.get("Woman", 0) or 0)
        prob_man = float(gender_obj.get("Man", 0) or 0)
        confidence = max(prob_woman, prob_man) * 100.0
        gender = "Woman" if prob_woman >= prob_man else "Man"

        return jsonify(
            {
                "ok": True,
                "gender": gender,
                "confidence": round(confidence, 1),
                "faceFound": True,
                "note": f"DeepFace: {gender} {round(confidence, 1)}%",
            }
        )
    except RuntimeError as e:
        return jsonify({"ok": False, "error": str(e)}), 500
    except Exception as e:  # noqa: BLE001
        return jsonify({"ok": False, "error": f"Unexpected error: {e}"}), 500


def _bytes_to_temp(raw: bytes) -> str:
    """Write bytes to a temp file so DeepFace can read them."""
    import tempfile

    fd, path = tempfile.mkstemp(suffix=".jpg")
    with os.fdopen(fd, "wb") as f:
        f.write(raw)
    return path


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port)
