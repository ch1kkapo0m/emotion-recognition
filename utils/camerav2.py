import cv2
import numpy as np
import tensorflow as tf
from collections import deque
import time

# налаштування
MODEL_PATH = 'models/best_model.keras'
EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
HISTORY_SECONDS = 30
FPS_UPDATE = 10

# кольори для кожної емоції (BGR)
EMOTION_COLORS = {
    'angry':    (0, 0, 220),
    'disgust':  (0, 140, 0),
    'fear':     (180, 0, 180),
    'happy':    (0, 220, 220),
    'neutral':  (180, 180, 180),
    'sad':      (220, 100, 0),
    'surprise': (0, 180, 220),
}

print("Завантаження моделі...")
model = tf.keras.models.load_model(MODEL_PATH)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
print("Готово!")

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# буфери для згладжування і графіку
SMOOTH_FRAMES = 10  
smooth_buffer = deque(maxlen=SMOOTH_FRAMES)

fps = cap.get(cv2.CAP_PROP_FPS) or 30
history_len = int(HISTORY_SECONDS * fps)
emotion_history = {e: deque(maxlen=history_len) for e in EMOTIONS}
session_counts = {e: 0 for e in EMOTIONS}

# FPS лічильник
fps_counter = 0
fps_display = 0
fps_start = time.time()

print("Камера запущена. Натисни Q щоб вийти, S щоб зберегти скріншот")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    fps_counter += 1
    if fps_counter % FPS_UPDATE == 0:
        fps_display = FPS_UPDATE / (time.time() - fps_start)
        fps_start = time.time()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48)
    )

    current_emotion = None
    current_probs = np.ones(7) / 7

    for (x, y, w, h) in faces:
        # підготовка зображення
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (48, 48))
        face = face.astype('float32') / 255.0
        face = np.expand_dims(face, axis=0)
        face = np.expand_dims(face, axis=-1)

        probs = model.predict(face, verbose=0)[0]

        # згладжування
        smooth_buffer.append(probs)
        smoothed = np.mean(smooth_buffer, axis=0)

        emotion_idx = np.argmax(smoothed)
        current_emotion = EMOTIONS[emotion_idx]
        current_probs = smoothed
        confidence = smoothed[emotion_idx] * 100

        # колір рамки
        color = EMOTION_COLORS[current_emotion]

        # рамка навколо обличчя
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

        # підпис над обличчям
        label = f"{current_emotion.upper()} {confidence:.1f}%"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        cv2.rectangle(frame, (x, y-35), (x + label_size[0] + 10, y), color, -1)
        cv2.putText(frame, label, (x+5, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

    # оновлення історії
    for i, emotion in enumerate(EMOTIONS):
        val = current_probs[i]
        emotion_history[emotion].append(val)
        if current_emotion == emotion:
            session_counts[emotion] += 1

    h_frame, w_frame = frame.shape[:2]

    # ===== ПАНЕЛЬ ЗЛІВА — bar chart емоцій =====
    panel_w = 220
    panel = np.zeros((h_frame, panel_w, 3), dtype=np.uint8)
    panel[:] = (30, 30, 30)

    cv2.putText(panel, "EMOTIONS", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)

    bar_max_w = 180
    for i, emotion in enumerate(EMOTIONS):
        y_pos = 55 + i * 70
        prob = current_probs[i]
        bar_w = int(prob * bar_max_w)
        color = EMOTION_COLORS[emotion]

        # фон бару
        cv2.rectangle(panel, (10, y_pos), (10 + bar_max_w, y_pos+25),
                      (60,60,60), -1)
        # бар
        if bar_w > 0:
            cv2.rectangle(panel, (10, y_pos), (10 + bar_w, y_pos+25),
                          color, -1)

        # назва емоції
        cv2.putText(panel, emotion, (10, y_pos+45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
        # відсоток
        cv2.putText(panel, f"{prob*100:.1f}%", (150, y_pos+45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # ===== ПАНЕЛЬ ЗНИЗУ — графік емоцій у часі =====
    graph_h = 160
    graph = np.zeros((graph_h, w_frame, 3), dtype=np.uint8)
    graph[:] = (20, 20, 20)

    cv2.putText(graph, f"EMOTION HISTORY ({HISTORY_SECONDS}s)",
                (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)

    for emotion in EMOTIONS:
        hist = list(emotion_history[emotion])
        if len(hist) < 2:
            continue
        color = EMOTION_COLORS[emotion]
        pts = []
        for j, val in enumerate(hist):
            px = int(j * w_frame / history_len)
            py = int(graph_h - 30 - val * (graph_h - 40))
            pts.append((px, py))

        for j in range(1, len(pts)):
            cv2.line(graph, pts[j-1], pts[j], color, 1)

    # легенда графіку
    for i, emotion in enumerate(EMOTIONS):
        x_leg = 10 + i * (w_frame // 7)
        cv2.rectangle(graph, (x_leg, graph_h-22),
                      (x_leg+12, graph_h-10), EMOTION_COLORS[emotion], -1)
        cv2.putText(graph, emotion[:3], (x_leg+15, graph_h-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180,180,180), 1)

    # ===== FPS і статистика =====
    cv2.putText(frame, f"FPS: {fps_display:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    total_frames = sum(session_counts.values())
    if total_frames > 0:
        dominant = max(session_counts, key=session_counts.get)
        dom_pct = session_counts[dominant] / total_frames * 100
        cv2.putText(frame, f"Session: {dominant} {dom_pct:.0f}%",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)

    # складання фінального кадру
    main_with_panel = np.hstack([panel, frame])
    # підганяємо ширину графіку
    graph_resized = cv2.resize(graph, (main_with_panel.shape[1], graph_h))
    final = np.vstack([main_with_panel, graph_resized])

    cv2.imshow('Emotion Recognition v2', final)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        filename = f"screenshot_{int(time.time())}.png"
        cv2.imwrite(filename, final)
        print(f"Збережено: {filename}")

# статистика сесії
print("\n=== СТАТИСТИКА СЕСІЇ ===")
total = sum(session_counts.values())
if total > 0:
    for emotion in EMOTIONS:
        pct = session_counts[emotion] / total * 100
        print(f"{emotion:10s}: {pct:.1f}%")

cap.release()
cv2.destroyAllWindows()