import os
import cv2
import numpy as np
from keras.models import load_model
import mediapipe as mp
import datetime
import time
from playsound import playsound
import pyttsx3
import speech_recognition as sr

# VOICE [SPEECH RECOGNITION AND MEDICATION REMINDER]
engine = pyttsx3.init()
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[1].id)
engine.setProperty('rate', 125)
r = sr.Recognizer()

# ALARM TIME [MEDICATION REMINDER]
alarm_time = "10:30:00"

# MEDIAPIPE BLAZEPOSE LANDMARKS [FALL DETECTION]
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
mp_drawing.DrawingSpec
detect = 0
stage = None

# CALCULATIONS OF ANGLE [FALL DETECTION]


def calculate_angle(a, b, c):
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # End

    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - \
        np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)

    if angle > 180.0:
        angle = 360-angle

    return angle


# MODEL LOADING [EMOTION RECOGNITION]
mod = load_model('C:/Users/Ryu/face/rawr/fernet_model.h5')
faceDetect = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
labels_dict = {0: 'Angry', 1: 'Disgust', 2: 'Fear',
               3: 'Happy', 4: 'Neutral', 5: 'Sad', 6: 'Surprise'}

# EMOTION RECOGNITION
video = cv2.VideoCapture(0)

while True:
    ret, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = faceDetect.detectMultiScale(gray, 1.3, 3)

    for x, y, w, h in faces:

        sub_face_img = gray[y:y+h, x:x+w]
        resized = cv2.resize(sub_face_img, (48, 48))
        normalize = resized/255.0
        reshaped = np.reshape(normalize, (1, 48, 48, 1))
        result = mod.predict(reshaped)
        result = mod.predict(reshaped)
        label = np.argmax(result, axis=1)[0]
        print(label)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 1)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 2)
        cv2.rectangle(frame, (x, y-40), (x+w, y), (50, 50, 255), -1)
        cv2.putText(frame, labels_dict[label], (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow("Emotion Recognition", frame)

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        image.flags.writeable = False
        results = pose.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        try:
            landmarks = results.pose_landmarks.landmark
            shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                   landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                    landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]

            angle = calculate_angle(shoulder, hip, knee)

            cv2.putText(image, str(angle), tuple(np.multiply(hip, [640, 480]).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

            if angle >= 140:
                stage = "stand"
            if angle < 120 and stage == 'stand':
                stage = "fall"
                engine.say("Fall Detected")
                engine.runAndWait()
                detect += 1

            with sr.Microphone() as source:
                print("Ready for Speech...")
                r.adjust_for_ambient_noise(source, duration=0.2)
                audio = r.listen(source)
                text = r.recognize_google(audio, language='en')
                text = text.lower()

                if "kitchen" in text.lower():
                    print("You said kitchen")
                    r = sr.Recognizer()
                    engine.say("Going to the kitchen")
                    engine.runAndWait()
                    continue

                elif "lobby" in text.lower():
                    print("You said lobby")
                    r = sr.Recognizer()
                    engine.say("Going to the lobby")
                    engine.runAndWait()
                    continue

                elif "garden" in text.lower():
                    print("You said garden")
                    r = sr.Recognizer()
                    engine.say("Going to the garden")
                    engine.runAndWait()
                    continue

        except sr.UnknownValueError:
            print("Can't Understand Audio...")

        except sr.RequestError as e:
            print("Could not request results; {0}".format(e))

        except:
            pass
        cv2.rectangle(image, (0, 0), (100, 70), (195, 144, 98), -1)
        cv2.putText(image, "Detect", (15, 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(image, stage, (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (255, 255, 255), 2, cv2.LINE_AA)
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS, mp_drawing.DrawingSpec(color=(
            245, 117, 66), thickness=2, circle_radius=2), mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))

        cv2.imshow('Fall Detection', image)

    current_time = datetime.datetime.now().strftime("%H:%M:%S")
    if current_time == alarm_time:
        print("Time to take your medication!")
        engine.say("Hello! It is time to take your medication!")
        engine.runAndWait()
    time.sleep(1)

    k = cv2.waitKey(1)
    if k == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
