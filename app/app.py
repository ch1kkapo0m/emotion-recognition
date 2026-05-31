import os
import socket
import numpy as np
import tensorflow as tf
import cv2
from flask import Flask, render_template, request, jsonify, Response
from PIL import Image
import base64
import io
import threading
import time
import qrcode

app = Flask(__name__)

EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
EMOTION_COLORS = {
    'angry':    '#DC2626',
    'disgust':  '#16A34A',
    'fear':     '#9333EA',
    'happy':    '#CA8A04',
    'neutral':  '#6B7280',
    'sad':      '#2563EB',
    'surprise': '#0891B2',
}
EMOTION_EMOJI = {
    'angry': '😠', 'disgust': '🤢', 'fear': '😨',
    'happy': '😊', 'neutral': '😐', 'sad': '😢', 'surprise': '😲'
}

print("Завантаження моделі...")
model = tf.keras.models.load_model('../models/best_model.keras')
face_cascade = cv2.CascadeClassifier('../haarcascade_frontalface_default.xml')
print("Готово!")

# ─── глобальні змінні для камери ───────────────────────────────────────────
camera_active = False
latest_emotion_data = {}


# ─── визначення локального IP ──────────────────────────────────────────────
def get_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return '127.0.0.1'


LOCAL_IP = get_local_ip()
PORT = 5000


# ─── допоміжна функція: передбачення для фото ──────────────────────────────
def predict_emotion(image_array):
    gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )
    if len(faces) == 0:
        return None

    results = []
    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (48, 48))
        face = face.astype('float32') / 255.0
        face = np.expand_dims(face, axis=0)
        face = np.expand_dims(face, axis=-1)

        probs = model.predict(face, verbose=0)[0]
        emotion_idx = np.argmax(probs)
        emotion = EMOTIONS[emotion_idx]
        confidence = float(probs[emotion_idx]) * 100
        all_probs = {EMOTIONS[i]: round(float(probs[i]) * 100, 1) for i in range(7)}

        results.append({
            'emotion': emotion,
            'confidence': round(confidence, 1),
            'all_probs': all_probs,
            'color': EMOTION_COLORS[emotion],
            'emoji': EMOTION_EMOJI[emotion],
            'bbox': {'x': int(x), 'y': int(y), 'w': int(w), 'h': int(h)}
        })
    return results


# ─── генератор кадрів для ПК камери ────────────────────────────────────────
def generate_frames():
    global camera_active, latest_emotion_data

    EMOTION_COLORS_BGR = {
        'angry':    (0, 0, 220),
        'disgust':  (0, 160, 0),
        'fear':     (180, 0, 180),
        'happy':    (0, 200, 200),
        'neutral':  (160, 160, 160),
        'sad':      (220, 80, 0),
        'surprise': (200, 160, 0),
    }

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    face_cascade_cam = cv2.CascadeClassifier('../haarcascade_frontalface_default.xml')
    smooth_buffer = []
    SMOOTH_FRAMES = 5

    while camera_active:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.05)
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade_cam.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48)
        )

        for (x, y, w, h) in faces:
            face_crop = gray[y:y+h, x:x+w]
            face_input = cv2.resize(face_crop, (48, 48))
            face_input = face_input.astype('float32') / 255.0
            face_input = np.expand_dims(face_input, axis=0)
            face_input = np.expand_dims(face_input, axis=-1)

            probs = model.predict(face_input, verbose=0)[0]
            smooth_buffer.append(probs)
            if len(smooth_buffer) > SMOOTH_FRAMES:
                smooth_buffer.pop(0)
            smoothed = np.mean(smooth_buffer, axis=0)

            emotion_idx = np.argmax(smoothed)
            emotion = EMOTIONS[emotion_idx]
            confidence = float(smoothed[emotion_idx]) * 100
            color = EMOTION_COLORS_BGR[emotion]

            latest_emotion_data = {
                'emotion': emotion,
                'confidence': confidence,
                'all_probs': {
                    EMOTIONS[i]: round(float(smoothed[i]) * 100, 1)
                    for i in range(7)
                }
            }

            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            label = f"{emotion} {confidence:.1f}%"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
            cv2.rectangle(frame, (x, y-35), (x + label_size[0] + 10, y), color, -1)
            cv2.putText(frame, label, (x+5, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        ret2, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
        if not ret2:
            continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

    cap.release()


# ─── маршрути ──────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/mobile')
def mobile():
    return render_template('mobile.html')


@app.route('/qr')
def qr_code():
    url = f'http://{LOCAL_IP}:{PORT}/mobile'
    img = qrcode.make(url)
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    buf.seek(0)
    return Response(buf.getvalue(), mimetype='image/png')


@app.route('/qr_url')
def qr_url():
    return jsonify({'url': f'http://{LOCAL_IP}:{PORT}/mobile'})


@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'Зображення не знайдено'}), 400

    file = request.files['image']
    img = Image.open(file.stream).convert('RGB')
    img_array = np.array(img)

    results = predict_emotion(img_array)
    if results is None:
        return jsonify({'error': 'Обличчя не знайдено на зображенні'}), 400

    img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    for r in results:
        x, y, w, h = r['bbox']['x'], r['bbox']['y'], r['bbox']['w'], r['bbox']['h']
        color_hex = r['color'].lstrip('#')
        color_rgb = tuple(int(color_hex[i:i+2], 16) for i in (0, 2, 4))
        color_bgr = (color_rgb[2], color_rgb[1], color_rgb[0])
        cv2.rectangle(img_cv, (x, y), (x+w, y+h), color_bgr, 3)
        label = f"{r['emotion']} {r['confidence']:.1f}%"
        cv2.putText(img_cv, label, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color_bgr, 2)

    img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    buffered = io.BytesIO()
    pil_img.save(buffered, format="JPEG", quality=90)
    img_b64 = base64.b64encode(buffered.getvalue()).decode()

    return jsonify({
        'results': results,
        'image': f'data:image/jpeg;base64,{img_b64}'
    })


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/camera/start', methods=['POST'])
def camera_start():
    global camera_active
    camera_active = True
    return jsonify({'status': 'started'})


@app.route('/camera/stop', methods=['POST'])
def camera_stop():
    global camera_active
    camera_active = False
    return jsonify({'status': 'stopped'})


@app.route('/current_emotion')
def current_emotion():
    return jsonify(latest_emotion_data)


@app.route('/analyze_frame', methods=['POST'])
def analyze_frame():
    data = request.get_json()
    if not data or 'frame' not in data:
        return jsonify({'error': 'Немає кадру'}), 400

    frame_data = data['frame'].split(',')[1]
    frame_bytes = base64.b64decode(frame_data)
    nparr = np.frombuffer(frame_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if frame is None:
        return jsonify({'error': 'Помилка декодування'}), 400

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )

    if len(faces) == 0:
        return jsonify({'faces': 0})

    x, y, w, h = faces[0]
    face = gray[y:y+h, x:x+w]
    face = cv2.resize(face, (48, 48))
    face = face.astype('float32') / 255.0
    face = np.expand_dims(face, axis=0)
    face = np.expand_dims(face, axis=-1)

    probs = model.predict(face, verbose=0)[0]
    emotion_idx = np.argmax(probs)
    emotion = EMOTIONS[emotion_idx]
    confidence = float(probs[emotion_idx]) * 100
    all_probs = {EMOTIONS[i]: round(float(probs[i]) * 100, 1) for i in range(7)}

    return jsonify({
        'faces': len(faces),
        'emotion': emotion,
        'confidence': round(confidence, 1),
        'all_probs': all_probs,
        'emoji': EMOTION_EMOJI[emotion],
        'color': EMOTION_COLORS[emotion]
    })


if __name__ == '__main__':
    print(f"\n🌐 Локальна адреса: http://{LOCAL_IP}:{PORT}")
    print(f"📱 Мобільна версія: http://{LOCAL_IP}:{PORT}/mobile")
    print(f"📷 QR-код: http://{LOCAL_IP}:{PORT}/qr\n")
    app.run(host='0.0.0.0', port=PORT, debug=True, ssl_context='adhoc')
