
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# === CONFIG ===
USE_MOBILENET = True   # 🔄 switch between CNN (False) and MobileNetV2 (True)
dataset_dir = 'C:/Users/karti/OneDrive/Desktop/C++/ML/asl_alphabet_dataset'
batch_size = 32
img_size = 224 if USE_MOBILENET else 96   # MobileNetV2 requires 224x224

# === Data Augmentation ===
data_gen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    validation_split=0.2
)

train_generator = data_gen.flow_from_directory(
    dataset_dir,
    target_size=(img_size, img_size),
    color_mode="rgb" if USE_MOBILENET else "grayscale",
    batch_size=batch_size,
    class_mode="categorical",
    subset="training"
)

val_generator = data_gen.flow_from_directory(
    dataset_dir,
    target_size=(img_size, img_size),
    color_mode="rgb" if USE_MOBILENET else "grayscale",
    batch_size=batch_size,
    class_mode="categorical",
    subset="validation"
)

# === Save labels order for webcam use ===
labels = list(train_generator.class_indices.keys())
with open("labels.pkl", "wb") as f:
    pickle.dump(labels, f)

num_classes = train_generator.num_classes
print(f"Detected {num_classes} classes: {train_generator.class_indices}")

# === Model Selection ===
if USE_MOBILENET:
    # Transfer Learning with MobileNetV2
    base_model = MobileNetV2(weights="imagenet", include_top=False,
                             input_shape=(img_size, img_size, 3))
    base_model.trainable = False  # Freeze backbone initially

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation="relu")(x)
    x = Dropout(0.5)(x)
    output = Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs=base_model.input, outputs=output)
else:
    # Custom CNN
    model = Sequential([
        Conv2D(32, (3, 3), activation="relu", input_shape=(img_size, img_size, 1)),
        BatchNormalization(),
        MaxPooling2D((2, 2)),

        Conv2D(64, (3, 3), activation="relu"),
        BatchNormalization(),
        MaxPooling2D((2, 2)),

        Conv2D(128, (3, 3), activation="relu"),
        BatchNormalization(),
        MaxPooling2D((2, 2)),

        Flatten(),
        Dense(256, activation="relu"),
        Dropout(0.5),
        Dense(num_classes, activation="softmax")
    ])

# === Compile ===
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# === Callbacks ===
checkpoint = ModelCheckpoint("asl_model.h5", save_best_only=True, monitor="val_accuracy", mode="max")
early_stop = EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=3, verbose=1)

# === Training ===
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=15 if USE_MOBILENET else 20,
    callbacks=[checkpoint, early_stop, reduce_lr]
)

# === Optional Fine-tuning for MobileNet ===
if USE_MOBILENET:
    print("\n Fine-tuning top layers of MobileNetV2...\n")
    base_model.trainable = True  # unfreeze backbone
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), 
                  loss="categorical_crossentropy", 
                  metrics=["accuracy"])
    history_fine = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=15,
        callbacks=[checkpoint, early_stop, reduce_lr]
    )

# === Plotting ===
plt.plot(history.history["accuracy"], label="Train Acc")
plt.plot(history.history["val_accuracy"], label="Val Acc")
if USE_MOBILENET and "history_fine" in locals():
    plt.plot(history_fine.history["accuracy"], label="Train Acc (Fine-tune)")
    plt.plot(history_fine.history["val_accuracy"], label="Val Acc (Fine-tune)")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.title("ASL Model Accuracy")
plt.show()
