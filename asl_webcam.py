# === asl_webcam.py ===

import cv2
import numpy as np
import tensorflow as tf
import pickle

# === Load trained model ===
model = tf.keras.models.load_model("asl_model.h5")

# === Load labels from training ===
labels = pickle.load(open("labels.pkl", "rb"))
print("Loaded labels:", labels)

# === Webcam Setup ===
cap = cv2.VideoCapture(0)
IMG_SIZE = 224   # must match training size (224 for MobileNet, 96 if CNN)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess frame
    img = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # ensure RGB
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)

    # Predict
    preds = model.predict(img, verbose=0)
    class_id = np.argmax(preds)
    confidence = np.max(preds)

    # Get label safely
    label = labels[class_id] if class_id < len(labels) else "?"

    # Display result
    text = f"{label} ({confidence*100:.1f}%)"
    cv2.putText(frame, text, (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("ASL Detection", frame)

    # Quit on 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
