import os
import cv2
import numpy as np
from PIL import Image
import mediapipe as mp

def create_dataset():
    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    if not cap.isOpened():
        print("Error: Could not open camera.")
        exit()
    count = 0
    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            count += 1
            cv2.imwrite("faceDataset/Aigul" + str(count) + ".jpg", gray[y:y+h, x:x+w])

        cv2.imshow('Face ID', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if count >= 300:
            break

    cap.release()
    cv2.destroyAllWindows()

def test_model():
    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("model.yml")
    if not cap.isOpened():
        print("Error: Could not open camera.")
        exit()

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
    mp_draw = mp.solutions.drawing_utils

    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        # Hand detection
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        finger_count = 0

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                finger_count = count_fingers(hand_landmarks)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            label, conf = recognizer.predict(gray[y:y+h, x:x+w])
            if conf < 50:
                if finger_count == 1:
                    put_text_on_video("Lan", frame, x, y)
                elif finger_count == 2:
                    put_text_on_video("Le", frame, x, y)
                else:
                    put_text_on_video("Lan", frame, x, y)
            else:
                put_text_on_video("Unknown", frame, x, y)

        cv2.imshow('Face ID', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def count_fingers(hand_landmarks):
    tip_ids = [4, 8, 12, 16, 20]
    fingers = []

    # Thumb
    if hand_landmarks.landmark[tip_ids[0]].x < hand_landmarks.landmark[tip_ids[0] - 1].x:
        fingers.append(1)
    else:
        fingers.append(0)

    # Other fingers
    for id in range(1, 5):
        if hand_landmarks.landmark[tip_ids[id]].y < hand_landmarks.landmark[tip_ids[id] - 2].y:
            fingers.append(1)
        else:
            fingers.append(0)

    return sum(fingers)

def put_text_on_video(text, frame, x, y):
    cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

def train_model(path):
    image_path = [os.path.join(path, f) for f in os.listdir(path)]
    faces = []
    labels = []
    for path in image_path:
        image = Image.open(path).convert('L')
        faces.append(np.array(image, 'uint8'))
        labels.append(1)
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    labels = np.array(labels)
    recognizer.train(faces, labels)
    recognizer.write("model.yml")

if __name__ == '__main__':
    #create_dataset()
    train_model("faceDataset")
    test_model()
