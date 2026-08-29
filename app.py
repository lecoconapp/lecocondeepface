# DeepFace gender-analysis server for Le Cocon's user verification.
#
# Endpoint:  POST /analyze
#   Accepts:  multipart/form-data with `image`, OR JSON {"image_base64": "..."}
#   Returns:  {"ok": true, "gender": "Woman"|"Man"|null, "confidence": <0-100>|null,
#              "faceFound": bool, "note": "...", "error": null}
#             or {"ok": false, "error": "..."}
#
# Run on the VPS with gunicorn:
#   ./venv/bin/gunicorn --workers 1 --threads 1 --timeout 600 -b 0.0.0.0:8000 app:app
#
# Memory notes (important on a 2-vCPU/4GB VPS):
#   - We cap TensorFlow to 2 threads and the OpenCV detector backend, which keeps
#     peak RAM low enough for the gender model (~537MB) to fit comfortably.
#   - gc.collect() frees memory between requests (few verifications/day = idle
#     in between, so this is effective).

import base64
import gc
import io
import os

# Limit TensorFlow/OpenMP CPU threads BEFORE tensorflow is imported, to cut peak
# memory use dramatically on a small VPS.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "2")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "2")

from flask import Flask, jsonify, request  # noqa: E402

app = Flask(__name__)

DETECTOR_BACKEND = "opencv"  # lightweight; avoids the heavier retinaface/mediapipe
DEEP_FACE = None


def _get_deepface():
    """Load DeepFace once (lazy singleton) and reuse it across requests."""
    global DEEP_FACE
    if DEEP_FACE is None:
        try:
            from deepface import DeepFace
            DEEP_FACE = DeepFace
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(f"Could not load DeepFace: {e}") from e
    return DEEP_FACE


def _load_image_bytes():
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

        try:
            results = df.analyze(
                img_path=image_path,
                actions=["gender"],
                enforce_detection=False,
                detector_backend=DETECTOR_BACKEND,
                silent=True,
            )
        except Exception as e:  # noqa: BLE001 - some versions raise when no face
            results = []

        # DeepFace <2023 returns a dict; newer returns a list.
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
        prob_woman = float(gender_obj.get("Woman", 0) or 0)
        prob_man = float(gender_obj.get("Man", 0) or 0)
        confidence = max(prob_woman, prob_man) * 100.0
        gender = "Woman" if prob_woman >= prob_man else "Man"

        response = {
            "ok": True,
            "gender": gender,
            "confidence": round(confidence, 1),
            "faceFound": True,
            "note": f"DeepFace: {gender} {round(confidence, 1)}%",
        }

        _cleanup(image_path)
        return jsonify(response)
    except RuntimeError as e:
        return jsonify({"ok": False, "error": str(e)}), 500
    except Exception as e:  # noqa: BLE001
        return jsonify({"ok": False, "error": f"Unexpected error: {e}"}), 500


def _bytes_to_temp(raw: bytes) -> str:
    import tempfile

    fd, path = tempfile.mkstemp(suffix=".jpg")
    with os.fdopen(fd, "wb") as f:
        f.write(raw)
    return path


def _cleanup(path: str | None):
    """Free temp file + run GC so peak memory stays low between requests."""
    if path:
        try:
            os.remove(path)
        except OSError:
            pass
    gc.collect()


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port)
