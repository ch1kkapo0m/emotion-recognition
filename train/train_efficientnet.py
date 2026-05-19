import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau
)

# =========================
# CONFIG
# =========================

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS_HEAD = 10
EPOCHS_FINE = 20
NUM_CLASSES = 7

TRAIN_DIR = 'data/train/train'
TEST_DIR = 'data/train/test'

# =========================
# DATA GENERATORS
# =========================

train_datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.15,
    height_shift_range=0.15,
    zoom_range=0.15,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
    validation_split=0.2,
    preprocessing_function=tf.keras.applications.efficientnet.preprocess_input
)

test_datagen = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.efficientnet.preprocess_input
)

train_gen = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    color_mode='rgb',
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

val_gen = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    color_mode='rgb',
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

test_gen = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    color_mode='rgb',
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

print("Train:", train_gen.samples)
print("Validation:", val_gen.samples)
print("Test:", test_gen.samples)

# =========================
# CLASS WEIGHTS
# =========================

total = train_gen.samples

class_counts = [3995, 436, 4097, 7215, 4965, 4830, 3171]

class_weights = {
    i: total / (NUM_CLASSES * c)
    for i, c in enumerate(class_counts)
}

print("Class weights:", class_weights)

# =========================
# MODEL
# =========================

base_model = tf.keras.applications.EfficientNetB0(
    include_top=False,
    weights='imagenet',
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

# freeze backbone
base_model.trainable = False

inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))

x = base_model(inputs, training=False)

x = layers.GlobalAveragePooling2D()(x)

x = layers.BatchNormalization()(x)

x = layers.Dropout(0.5)(x)

x = layers.Dense(
    256,
    activation='relu',
    kernel_regularizer=tf.keras.regularizers.l2(1e-4)
)(x)

x = layers.Dropout(0.4)(x)

outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)

model = tf.keras.Model(inputs, outputs)

# =========================
# COMPILE HEAD TRAINING
# =========================

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# =========================
# CALLBACKS
# =========================

os.makedirs('models', exist_ok=True)

callbacks = [
    ModelCheckpoint(
        'models/best_efficientnet.keras',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    ),

    EarlyStopping(
        monitor='val_accuracy',
        patience=8,
        restore_best_weights=True,
        verbose=1
    ),

    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1
    )
]

# =========================
# TRAIN HEAD
# =========================

print("\n=========================")
print("TRAINING CLASSIFIER HEAD")
print("=========================\n")

history_head = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS_HEAD,
    callbacks=callbacks,
    class_weight=class_weights
)

# =========================
# FINE TUNING
# =========================

print("\n=========================")
print("FINE TUNING")
print("=========================\n")

# unfreeze last layers
base_model.trainable = True

# freeze first layers
for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history_fine = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS_FINE,
    callbacks=callbacks,
    class_weight=class_weights
)

# =========================
# TEST EVALUATION
# =========================

print("\n=========================")
print("TEST EVALUATION")
print("=========================\n")

test_loss, test_acc = model.evaluate(test_gen)

print(f"Test Accuracy: {test_acc:.4f}")
print(f"Test Loss: {test_loss:.4f}")

# =========================
# PLOTS
# =========================

acc = (
    history_head.history['accuracy'] +
    history_fine.history['accuracy']
)

val_acc = (
    history_head.history['val_accuracy'] +
    history_fine.history['val_accuracy']
)

loss = (
    history_head.history['loss'] +
    history_fine.history['loss']
)

val_loss = (
    history_head.history['val_loss'] +
    history_fine.history['val_loss']
)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(acc, label='train_accuracy')
plt.plot(val_acc, label='val_accuracy')
plt.title('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(loss, label='train_loss')
plt.plot(val_loss, label='val_loss')
plt.title('Loss')
plt.legend()

plt.savefig('models/training_history.png')

print("Saved plots: models/training_history.png")

# =========================
# SAVE FINAL MODEL
# =========================

model.save('models/final_model.keras')

print("Model saved: models/final_model.keras")