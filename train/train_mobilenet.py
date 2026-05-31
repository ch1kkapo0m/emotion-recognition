import os
import logging
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.applications import MobileNetV2
from datetime import datetime

IMG_SIZE = 128
BATCH_SIZE = 64
EPOCHS_PHASE1 = 20   
EPOCHS_PHASE2 = 30
NUM_CLASSES = 7
TRAIN_DIR = 'data/train/train'
TEST_DIR = 'data/train/test'
EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

os.makedirs('logs', exist_ok=True)
os.makedirs('models', exist_ok=True)
log_filename = f"logs/mobilenet_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s — %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler(log_filename, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
log = logging.getLogger()

log.info("=" * 55)
log.info("НАВЧАННЯ MobileNetV2 — РОЗПІЗНАВАННЯ ЕМОЦІЙ")
log.info("=" * 55)
log.info(f"Дата і час запуску: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
log.info(f"IMG_SIZE: {IMG_SIZE}x{IMG_SIZE} RGB")
log.info(f"BATCH_SIZE: {BATCH_SIZE}")
log.info(f"EPOCHS_PHASE1: {EPOCHS_PHASE1}")
log.info(f"EPOCHS_PHASE2: {EPOCHS_PHASE2}")
log.info(f"NUM_CLASSES: {NUM_CLASSES}")
log.info(f"EMOTIONS: {EMOTIONS}")

log.info("-" * 55)
log.info("Ініціалізація генераторів даних...")

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    zoom_range=0.1,
    validation_split=0.2
)
test_datagen = ImageDataGenerator(rescale=1./255)

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

log.info(f"Train зображень: {train_gen.samples}")
log.info(f"Val зображень:   {val_gen.samples}")
log.info(f"Test зображень:  {test_gen.samples}")
log.info(f"Класи: {train_gen.class_indices}")

# базова модель
log.info("-" * 55)
log.info("Завантаження MobileNetV2 (ImageNet weights)...")

base_model = MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False

log.info(f"Базова модель завантажена: {len(base_model.layers)} шарів")
log.info("Всі шари заморожені")

# архітектура
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.BatchNormalization(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(NUM_CLASSES, activation='softmax')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

total_params = model.count_params()
trainable_params = sum([tf.size(w).numpy() for w in model.trainable_weights])
log.info(f"Загальна кількість параметрів: {total_params:,}")
log.info(f"Параметрів для навчання: {trainable_params:,}")

# дисбаланс класів
class_counts = [3995, 436, 4097, 7215, 4965, 4830, 3171]
total = train_gen.samples
class_weights = {i: total / (NUM_CLASSES * c) for i, c in enumerate(class_counts)}

log.info("-" * 55)
log.info("Ваги класів (компенсація дисбалансу):")
for i, (emotion, weight) in enumerate(zip(EMOTIONS, class_weights.values())):
    log.info(f"  {emotion}: {weight:.4f}  ({class_counts[i]} фото)")

# ============================================================
# ФАЗА 1
# ============================================================
log.info("=" * 55)
log.info("ФАЗА 1 — навчання верхніх шарів")
log.info(f"Learning rate: 0.001")
log.info(f"Заморожено: {len(base_model.layers)} шарів MobileNetV2")
log.info("=" * 55)

start_phase1 = datetime.now()

callbacks_phase1 = [
    ModelCheckpoint(
        'models/mobilenet_phase1.keras',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    ),
    EarlyStopping(
        monitor='val_accuracy',
        patience=5,
        restore_best_weights=True,
        verbose=1
    )
]

history1 = model.fit(
    train_gen,
    epochs=EPOCHS_PHASE1,
    validation_data=val_gen,
    callbacks=callbacks_phase1,
    class_weight=class_weights
)

end_phase1 = datetime.now()
duration_phase1 = end_phase1 - start_phase1

best_val_acc1 = max(history1.history['val_accuracy'])
best_epoch1 = history1.history['val_accuracy'].index(best_val_acc1) + 1

log.info("-" * 55)
log.info("ФАЗА 1 — РЕЗУЛЬТАТИ:")
log.info(f"  Епох пройдено: {len(history1.history['accuracy'])}")
log.info(f"  Найкраща val_accuracy: {best_val_acc1:.4f} (епоха {best_epoch1})")
log.info(f"  Найменша val_loss: {min(history1.history['val_loss']):.4f}")
log.info(f"  Фінальна train_accuracy: {history1.history['accuracy'][-1]:.4f}")
log.info(f"  Тривалість: {str(duration_phase1).split('.')[0]}")

# ============================================================
# ФАЗА 2
# ============================================================
log.info("=" * 55)
log.info("ФАЗА 2 — fine-tuning")
log.info(f"Learning rate: 0.00001")

base_model.trainable = True
fine_tune_at = len(base_model.layers) - 30

for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

log.info(f"Розморожено шарів: {len(base_model.layers) - fine_tune_at}")
log.info(f"Заморожено шарів:  {fine_tune_at}")
log.info("=" * 55)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

start_phase2 = datetime.now()

callbacks_phase2 = [
    ModelCheckpoint(
        'models/mobilenet_best.keras',
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
        patience=4,
        min_lr=1e-7,
        verbose=1
    )
]

history2 = model.fit(
    train_gen,
    epochs=EPOCHS_PHASE2,
    validation_data=val_gen,
    callbacks=callbacks_phase2,
    class_weight=class_weights
)

end_phase2 = datetime.now()
duration_phase2 = end_phase2 - start_phase2

best_val_acc2 = max(history2.history['val_accuracy'])
best_epoch2 = history2.history['val_accuracy'].index(best_val_acc2) + 1

log.info("-" * 55)
log.info("ФАЗА 2 — РЕЗУЛЬТАТИ:")
log.info(f"  Епох пройдено: {len(history2.history['accuracy'])}")
log.info(f"  Найкраща val_accuracy: {best_val_acc2:.4f} (епоха {best_epoch2})")
log.info(f"  Найменша val_loss: {min(history2.history['val_loss']):.4f}")
log.info(f"  Фінальна train_accuracy: {history2.history['accuracy'][-1]:.4f}")
log.info(f"  Тривалість: {str(duration_phase2).split('.')[0]}")

# графіки
log.info("-" * 55)
log.info("Збереження графіків...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

axes[0,0].plot(history1.history['accuracy'], label='train')
axes[0,0].plot(history1.history['val_accuracy'], label='val')
axes[0,0].set_title('Фаза 1 — Accuracy')
axes[0,0].legend()

axes[0,1].plot(history1.history['loss'], label='train')
axes[0,1].plot(history1.history['val_loss'], label='val')
axes[0,1].set_title('Фаза 1 — Loss')
axes[0,1].legend()

axes[1,0].plot(history2.history['accuracy'], label='train')
axes[1,0].plot(history2.history['val_accuracy'], label='val')
axes[1,0].set_title('Фаза 2 — Accuracy')
axes[1,0].legend()

axes[1,1].plot(history2.history['loss'], label='train')
axes[1,1].plot(history2.history['val_loss'], label='val')
axes[1,1].set_title('Фаза 2 — Loss')
axes[1,1].legend()

plt.suptitle('MobileNetV2 — Навчання', fontsize=14)
plt.tight_layout()
plt.savefig('models/mobilenet_history.png')

# підсумок
total_duration = end_phase2 - start_phase1
log.info("=" * 55)
log.info("ПІДСУМОК НАВЧАННЯ")
log.info("=" * 55)
log.info(f"Датасет:              FER2013")
log.info(f"Train:                {train_gen.samples} зображень")
log.info(f"Val:                  {val_gen.samples} зображень")
log.info(f"Test:                 {test_gen.samples} зображень")
log.info(f"Розмір зображення:    {IMG_SIZE}x{IMG_SIZE} RGB")
log.info(f"Розмір батчу:         {BATCH_SIZE}")
log.info(f"---")
log.info(f"Фаза 1 — val_accuracy: {best_val_acc1:.4f}")
log.info(f"Фаза 2 — val_accuracy: {best_val_acc2:.4f}")
log.info(f"Покращення:           +{(best_val_acc2 - best_val_acc1)*100:.2f}%")
log.info(f"---")
log.info(f"Загальний час:        {str(total_duration).split('.')[0]}")
log.info(f"Модель збережена:     models/mobilenet_best.keras")
log.info(f"Лог збережено:        {log_filename}")
log.info("=" * 55)