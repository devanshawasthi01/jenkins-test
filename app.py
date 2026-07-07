import hashlib
import json
import os
import time
import urllib.request
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from flask import Flask, Response, jsonify, render_template, request
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

app = Flask(__name__)

# Runtime configuration
MODEL_PATH = os.getenv("MODEL_PATH", "blaze_face_short_range.tflite")
MIN_DETECTION_CONFIDENCE = float(os.getenv("MIN_DETECTION_CONFIDENCE", "0.5"))
CAMERA_INDEX = int(os.getenv("CAMERA_INDEX", "0"))
APP_PORT = int(os.getenv("APP_PORT", "5000"))
APP_DEBUG = os.getenv("APP_DEBUG", "false").lower() == "true"
ENABLE_RETRAIN_CAPTURE = os.getenv("ENABLE_RETRAIN_CAPTURE", "false").lower() == "true"
CAPTURE_EVERY_N_FRAMES = int(os.getenv("CAPTURE_EVERY_N_FRAMES", "30"))

# Pretrained BlazeFace model from MediaPipe official CDN
MODEL_URL = os.getenv(
    "MODEL_URL",
    "https://storage.googleapis.com/mediapipe-models/"
    "face_detector/blaze_face_short_range/float16/1/"
    "blaze_face_short_range.tflite",
)

START_TIME = time.time()
FRAME_COUNTER = 0

# Operational metrics (Prometheus)
STREAMS_STARTED = Counter("streams_started_total", "Total started MJPEG streams")
FRAMES_PROCESSED = Counter("frames_processed_total", "Total processed video frames")
FACES_DETECTED = Counter("faces_detected_total", "Total detected faces across frames")
ACTIVE_STREAMS = Gauge("active_streams", "Current number of active streams")
INFERENCE_LATENCY_SECONDS = Histogram(
    "inference_latency_seconds",
    "Face detector inference latency in seconds",
    buckets=(0.005, 0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 1.0),
)
MODEL_INFO = Gauge(
    "model_info",
    "Model info with labels set to 1",
    labelnames=("model_name", "model_sha256", "min_confidence"),
)

os.makedirs("logs", exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs(os.path.join("data", "retrain", "raw"), exist_ok=True)
INFERENCE_LOG_FILE = os.path.join("logs", "inference_events.jsonl")
FEEDBACK_LOG_FILE = os.path.join("logs", "feedback_labels.jsonl")
RETRAIN_SAMPLES_LOG_FILE = os.path.join("logs", "retrain_samples.jsonl")
CALIBRATOR_PATH = os.path.join("models", "confidence_calibrator.json")
MODEL_ARTIFACT_PATH = os.path.join("artifacts", "model.plk")

if not os.path.exists(MODEL_PATH):
    print("Downloading BlazeFace model (~800KB)...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Model downloaded.")


def _file_sha256(path):
    sha = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(8192), b""):
            sha.update(chunk)
    return sha.hexdigest()


MODEL_SHA256 = _file_sha256(MODEL_PATH)
MODEL_NAME = "blaze_face_short_range"
MODEL_INFO.labels(
    model_name=MODEL_NAME,
    model_sha256=MODEL_SHA256,
    min_confidence=str(MIN_DETECTION_CONFIDENCE),
).set(1)

# Build the MediaPipe Tasks face detector (new API)
_base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
_options = mp_vision.FaceDetectorOptions(
    base_options=_base_options, min_detection_confidence=MIN_DETECTION_CONFIDENCE
)
detector = mp_vision.FaceDetector.create_from_options(_options)

calibrator = None
if os.path.exists(CALIBRATOR_PATH):
    with open(CALIBRATOR_PATH, "r", encoding="utf-8") as file:
        calibrator = json.load(file)
    print(f"Loaded calibrator model from {CALIBRATOR_PATH}")


def _calibrate_confidence(raw_score):
    if calibrator is None:
        return raw_score

    bins = int(calibrator.get("bins", 20))
    probabilities = calibrator.get("probabilities", [])
    if not probabilities:
        return raw_score

    clipped = min(max(float(raw_score), 0.0), 1.0)
    index = min(int(clipped * bins), bins - 1)
    return float(probabilities[index])


def generate_frames():
    global FRAME_COUNTER

    camera = cv2.VideoCapture(CAMERA_INDEX)
    STREAMS_STARTED.inc()
    ACTIVE_STREAMS.inc()
    try:
        while True:
            success, frame = camera.read()
            if not success:
                break

            # MediaPipe requires RGB input
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            inference_start = time.perf_counter()
            result = detector.detect(mp_image)
            INFERENCE_LATENCY_SECONDS.observe(time.perf_counter() - inference_start)

            face_count = 0
            if result.detections:
                face_count = len(result.detections)
                for detection in result.detections:
                    bbox = detection.bounding_box
                    x = max(0, bbox.origin_x)
                    y = max(0, bbox.origin_y)
                    bw = bbox.width
                    bh = bbox.height

                    # Draw bounding box
                    cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 255, 0), 2)

                    # Show calibrated confidence score (if calibrator is available)
                    raw_confidence = float(detection.categories[0].score)
                    confidence = _calibrate_confidence(raw_confidence)
                    cv2.putText(
                        frame,
                        f"{confidence * 100:.1f}% (raw {raw_confidence * 100:.1f}%)",
                        (x, max(y - 10, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        1,
                    )

            FRAME_COUNTER += 1
            FRAMES_PROCESSED.inc()
            FACES_DETECTED.inc(face_count)

            event = {
                "timestamp": int(time.time()),
                "frame_id": FRAME_COUNTER,
                "faces": face_count,
                "model": MODEL_NAME,
                "model_sha256": MODEL_SHA256,
            }
            with open(INFERENCE_LOG_FILE, "a", encoding="utf-8") as log_file:
                log_file.write(json.dumps(event) + "\n")

            if ENABLE_RETRAIN_CAPTURE and FRAME_COUNTER % CAPTURE_EVERY_N_FRAMES == 0:
                image_name = f"frame_{FRAME_COUNTER:08d}.jpg"
                image_path = os.path.join("data", "retrain", "raw", image_name)
                cv2.imwrite(image_path, frame)
                sample = {
                    "timestamp": int(time.time()),
                    "frame_id": FRAME_COUNTER,
                    "image_path": image_path,
                    "faces": face_count,
                }
                with open(RETRAIN_SAMPLES_LOG_FILE, "a", encoding="utf-8") as log_file:
                    log_file.write(json.dumps(sample) + "\n")

            # Face count
            cv2.putText(
                frame,
                f"Faces: {face_count}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
            )

            # Model label
            cv2.putText(
                frame,
                "Model: BlazeFace (MediaPipe Tasks)",
                (10, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 0),
                1,
            )

            _, buffer = cv2.imencode(".jpg", frame)
            frame_bytes = buffer.tobytes()
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
            )
    finally:
        camera.release()
        ACTIVE_STREAMS.dec()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/health")
def health():
    return jsonify(
        {
            "status": "ok",
            "uptime_seconds": int(time.time() - START_TIME),
            "model_path": MODEL_PATH,
            "model_sha256": MODEL_SHA256,
            "min_detection_confidence": MIN_DETECTION_CONFIDENCE,
            "camera_index": CAMERA_INDEX,
        }
    )


@app.route("/metrics")
def metrics():
    return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)


@app.route("/feedback", methods=["POST"])
def feedback():
    payload = request.get_json(silent=True) or {}
    raw_confidence = payload.get("raw_confidence")
    label = payload.get("label")

    if raw_confidence is None or label is None:
        return jsonify({"error": "raw_confidence and label are required"}), 400

    try:
        raw_confidence = float(raw_confidence)
        label = int(label)
    except (TypeError, ValueError):
        return jsonify({"error": "raw_confidence must be float and label must be int"}), 400

    if label not in (0, 1):
        return jsonify({"error": "label must be 0 (false positive) or 1 (true face)"}), 400

    record = {
        "timestamp": int(time.time()),
        "raw_confidence": raw_confidence,
        "label": label,
    }
    with open(FEEDBACK_LOG_FILE, "a", encoding="utf-8") as log_file:
        log_file.write(json.dumps(record) + "\n")

    return jsonify({"status": "accepted", "record": record})


@app.route("/mlops/status")
def mlops_status():
    feedback_records = 0
    if os.path.exists(FEEDBACK_LOG_FILE):
        with open(FEEDBACK_LOG_FILE, "r", encoding="utf-8") as file:
            feedback_records = sum(1 for line in file if line.strip())

    return jsonify(
        {
            "calibrator_loaded": calibrator is not None,
            "calibrator_path": CALIBRATOR_PATH,
            "model_artifact_path": MODEL_ARTIFACT_PATH,
            "model_artifact_exists": os.path.exists(MODEL_ARTIFACT_PATH),
            "feedback_log": FEEDBACK_LOG_FILE,
            "feedback_records": feedback_records,
            "capture_enabled": ENABLE_RETRAIN_CAPTURE,
            "capture_every_n_frames": CAPTURE_EVERY_N_FRAMES,
        }
    )


if __name__ == "__main__":
    app.run(debug=APP_DEBUG, host="0.0.0.0", port=APP_PORT)
