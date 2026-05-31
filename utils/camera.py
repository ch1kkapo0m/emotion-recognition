import cv2
import numpy as np
import tensorflow as tf

EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
model = tf.keras.models.load_model('models/best_model.keras')
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


cap = cv2.VideoCapture(0)
print("Камера запущена. Натисни Q щоб вийти")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48)
    )

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (48, 48))
        face = face.astype('float32') / 255.0
        face = np.expand_dims(face, axis=0)
        face = np.expand_dims(face, axis=-1)

        predictions = model.predict(face, verbose=0)
        emotion_idx = np.argmax(predictions)
        emotion = EMOTIONS[emotion_idx]
        confidence = predictions[0][emotion_idx] * 100

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        label = f"{emotion} {confidence:.1f}%"
        cv2.putText(frame, label, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        for i, (em, prob) in enumerate(zip(EMOTIONS, predictions[0])):
            bar_w = int(prob * 150)
            cv2.rectangle(frame, (10, 10 + i*25),
                         (10 + bar_w, 30 + i*25), (0, 255, 0), -1)
            cv2.putText(frame, f"{em}: {prob*100:.1f}%",
                       (170, 27 + i*25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imshow('Emotion Recognition', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()