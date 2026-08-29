# Le Cocon gender-verification server (lightweight).
#
# Replaces DeepFace with a tiny ONNX gender classifier + OpenCV DNN face
# detector. Peak RAM ~500MB-1GB -> fits the 4GB Hetzner VPS comfortably.
#
# Endpoint: POST /analyze
#   Accepts: multipart/form-data with `image`, OR JSON {"image_base64": "..."}
#   Returns: {"ok": true, "gender": "Woman"|"Man"|null, "confidence": <0-100>|null,
#             "faceFound": bool, "note": "...", "error": null}
#
# Run: ./venv/bin/gunicorn --workers 1 --threads 1 --timeout 120 -b 0.0.0.0:8000 app:app

import base64
import os
import tempfile

import cv2
import numpy as np
import onnxruntime as ort
from flask import Flask, jsonify, request

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

FACE_PROTO = os.path.join(MODELS_DIR, "deploy.prototxt")
FACE_MODEL = os.path.join(MODELS_DIR, "res10.caffemodel")
GENDER_MODEL = os.path.join(MODELS_DIR, "gender.onnx")

FACE_SIZE = 300
CONFIDENCE_THRESHOLD = 0.7  # pct for a valid face detection
MIN_FACE = 40  # min face width/height (px) to accept

app = Flask(__name__)

_face_net = None
_gender_sess = None


def _get_face_net():
    global _face_net
    if _face_net is None:
        _face_net = cv2.dnn.readNetFromCaffe(FACE_PROTO, FACE_MODEL)
    return _face_net


def _get_gender_session():
    global _gender_sess
    if _gender_sess is None:
        _gender_sess = ort.InferenceSession(
            GENDER_MODEL, providers=["CPUExecutionProvider"]
        )
    return _gender_sess


def _detect_faces(img_bgr):
    """Return list of (x1,y1,x2,y2) face boxes for the largest/most confident faces."""
    net = _get_face_net()
    h, w = img_bgr.shape[:2]
    blob = cv2.dnn.blobFromImage(
        cv2.resize(img_bgr, (FACE_SIZE, FACE_SIZE)),
        scalefactor=1.0,
        size=(FACE_SIZE, FACE_SIZE),
        mean=(104.0, 177.0, 123.0),
        swapRB=False,
        crop=False,
    )
    net.setInput(blob)
    detections = net.forward()
    faces = []
    for i in range(detections.shape[2]):
        confidence = float(detections[0, 0, i, 2])
        if confidence < CONFIDENCE_THRESHOLD:
            continue
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        x1, y1, x2, y2 = box.astype("int")
        if (x2 - x1) < MIN_FACE or (y2 - y1) < MIN_FACE:
            continue
        faces.append((x1, y1, x2, y2, confidence))
    # Highest confidence first
    faces.sort(key=lambda f: f[4], reverse=True)
    return faces


def _gender_from_face(img_bgr, box):
    """Run the ONNX ViT gender model on a cropped face. Returns (gender, prob)."""
    x1, y1, x2, y2 = box[:4]
    face = img_bgr[y1:y2, x1:x2]
    if face.size == 0:
        return None, 0.0
    rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_CUBIC)
    # ViT preprocessing: /255 then (x-0.5)/0.5  -> range [-1,1]
    x = np.asarray(resized, dtype=np.float32) / 255.0
    x = (x - 0.5) / 0.5
    x = np.transpose(x, (2, 0, 1))  # HWC -> CHW
    x = np.expand_dims(x, axis=0).astype(np.float32)  # 1,3,224,224

    sess = _get_gender_session()
    logits = sess.run(None, {"pixel_values": x})[0][0]
    exp = np.exp(logits - np.max(logits))
    probs = exp / exp.sum()
    # id2label: 0=female, 1=male
    prob_female, prob_male = float(probs[0]), float(probs[1])
    gender = "Woman" if prob_female >= prob_male else "Man"
    confidence = max(prob_female, prob_male) * 100.0
    return gender, confidence


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
    return jsonify({"ok": True, "service": "lecocon-verify"})


@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        raw = _load_image_bytes()
        if not raw:
            return jsonify(
                {"ok": False, "error": "No image provided (multipart 'image' or JSON 'image_base64')."}
            ), 400

        fd, path = tempfile.mkstemp(suffix=".jpg")
        with os.fdopen(fd, "wb") as f:
            f.write(raw)

        try:
            img = cv2.imread(path)
            if img is None:
                return jsonify({"ok": False, "error": "Could not decode image"}), 400

            faces = _detect_faces(img)
            if not faces:
                return jsonify(
                    {
                        "ok": True,
                        "gender": None,
                        "confidence": None,
                        "faceFound": False,
                        "note": "No clear face detected.",
                    }
                )

            gender, confidence = _gender_from_face(img, faces[0])
            if gender is None:
                return jsonify(
                    {
                        "ok": True,
                        "gender": None,
                        "confidence": None,
                        "faceFound": True,
                        "note": "Face found but could not classify.",
                    }
                )

            return jsonify(
                {
                    "ok": True,
                    "gender": gender,
                    "confidence": round(confidence, 1),
                    "faceFound": True,
                    "note": f"Verify: {gender} {round(confidence, 1)}%",
                }
            )
        finally:
            try:
                os.remove(path)
            except OSError:
                pass
    except Exception as e:  # noqa: BLE001
        return jsonify({"ok": False, "error": f"Unexpected error: {e}"}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port)
