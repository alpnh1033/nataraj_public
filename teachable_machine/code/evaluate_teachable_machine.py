# Load a Teachable Machine model and evaluate it on testing and cross-testing sets.

import numpy as np
import tensorflow as tf
from PIL import Image
from pathlib import Path

#  Paths to the trained model and the dataset root.

MODEL_PATH = Path("../model/teachable_machine.h5")
DATASET_ROOT = Path("../../../dataset/mudras/final-dataset")

# Ordered class labels used by the model. The index order must match the model output.
LABELS = [
    "Kartari Mukha",
    "mukula",
    "mushti",
    "Pataaka",
    "Shikhara",
    "Suchi",
    "Tripataka",
]

# Load the trained model once so the evaluation loop can reuse it.
model = tf.keras.models.load_model(MODEL_PATH)

# Preprocess an image to the format expected by the model: resize to 224x224 and normalize pixel values to [-1, 1].
def preprocess_image(image_path):
    """Open an image, resize it to MobileNet input size, and normalize to [-1, 1]."""
    with Image.open(image_path) as img:
        img = img.convert("RGB").resize((224, 224))

    arr = np.asarray(img, dtype=np.float32)
    return arr / 127.5 - 1.0

# Predict the class label for a single image using the loaded model.
def predict(image_path):
    """Return the predicted class label for a single image."""
    image_batch = preprocess_image(image_path)[None, ...]
    probabilities = model.predict(image_batch, verbose=0)[0]
    predicted_index = int(np.argmax(probabilities))
    return LABELS[predicted_index]

# Score all images in a dataset split and return the number of correct predictions and the total number of images.
def score(split_name):
    """Score all images in a dataset split and return (correct, total)."""
    correct = 0
    total = 0

    # Iterate each mudra class and then each image in that split folder.
    # .title() because LABELS has "mukula"/"mushti" lowercase (how Teachable
    # Machine saved them) but the dataset folders are capitalized -- matters
    # on case-sensitive filesystems (Linux/Mac/Colab), not just Windows.
    for label in LABELS:
        split_dir = DATASET_ROOT / label.title() / split_name
        for photo in split_dir.glob("*.*"):
            total += 1
            correct += predict(photo) == label

    return correct, total


# Evaluate both the same-source validation set and the cross-source set.
for split in ("testing", "cross-testing"):
    correct, total = score(split)
    print(f"{split}: {correct}/{total} = {correct / total:.1%}")
