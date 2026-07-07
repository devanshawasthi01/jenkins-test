import json
import os
import pickle
from collections import Counter

FEEDBACK_LOG_FILE = os.path.join("logs", "feedback_labels.jsonl")
MODEL_OUTPUT_PATH = os.path.join("models", "confidence_calibrator.json")
MODEL_ARTIFACT_PATH = os.path.join("artifacts", "model.plk")
NUM_BINS = 20


def load_feedback(path):
    features = []
    labels = []

    if not os.path.exists(path):
        raise FileNotFoundError(f"Feedback file not found: {path}")

    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            raw_confidence = float(row["raw_confidence"])
            label = int(row["label"])
            if label not in (0, 1):
                continue
            features.append([raw_confidence])
            labels.append(label)

    return features, labels


def train_bin_calibrator(features, labels, bins=NUM_BINS):
    counts = [0] * bins
    positives = [0] * bins

    for feature, label in zip(features, labels):
        score = min(max(float(feature[0]), 0.0), 1.0)
        index = min(int(score * bins), bins - 1)
        counts[index] += 1
        if label == 1:
            positives[index] += 1

    probabilities = []
    for idx in range(bins):
        # Laplace smoothing keeps probabilities stable for low-sample bins.
        probability = (positives[idx] + 1.0) / (counts[idx] + 2.0)
        probabilities.append(probability)

    return {
        "type": "bin_calibrator",
        "bins": bins,
        "probabilities": probabilities,
        "counts": counts,
    }


def main():
    x, y = load_feedback(FEEDBACK_LOG_FILE)

    if len(y) < 20:
        raise ValueError("Need at least 20 feedback rows before retraining.")

    class_counts = Counter(y)
    if len(class_counts) < 2:
        raise ValueError("Need both classes (0 and 1) in feedback for training.")

    model = train_bin_calibrator(x, y)

    os.makedirs(os.path.dirname(MODEL_OUTPUT_PATH), exist_ok=True)
    with open(MODEL_OUTPUT_PATH, "w", encoding="utf-8") as file:
        json.dump(model, file, indent=2)

    os.makedirs(os.path.dirname(MODEL_ARTIFACT_PATH), exist_ok=True)
    with open(MODEL_ARTIFACT_PATH, "wb") as file:
        pickle.dump(model, file)

    print("Calibrator model trained and saved.")
    print(f"Output: {MODEL_OUTPUT_PATH}")
    print(f"Artifact: {MODEL_ARTIFACT_PATH}")
    print(f"Rows used: {len(y)}")
    print(f"Class balance: {dict(class_counts)}")


if __name__ == "__main__":
    main()
