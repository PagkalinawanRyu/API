from tkinter import *
from PIL import Image, ImageTk
import speech_recognition as sr
import cv2
import numpy as np
from keras.models import load_model
import mediapipe as mp
import datetime
import time
import pyttsx3
from pygame import mixer

root = Tk()
root.title('Features')
root.minsize(700,630)
root.maxsize(700,630)
root.config(bg='#58F')

#features
alarm_time = "10:30:00"
engine = pyttsx3.init()
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[1].id)
engine.setProperty('rate', 125)
r = sr.Recognizer()
mixer.init()
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
mp_drawing.DrawingSpec
detect = 0
stage = None

def calculate_angle(a, b, c):
    a = np.array(a) 
    b = np.array(b)  
    c = np.array(c)  

    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - \
        np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)

    if angle > 180.0:
        angle = 360-angle

    return angle

mod = load_model('C:/Users/Ryu/face/musicmediapipe/fernetmod.h5')
faceDetect = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
labels_dict = {0: 'Angry', 1: 'Happy', 2: 'Sad'}

video = cv2.VideoCapture(0)
video.set(cv2.CAP_PROP_BUFFERSIZE, 2)

if(video.isOpened() == False):
    print('Unable to read camera feed')

def exitWindow():
    video.release()
    cv2.destroyAllWindows()
    root.destroy()
    root.quit()

def map():
    pass

def getrobot():
    pass

f1 = LabelFrame(root, background='red')
f1.pack()
l1 = Label(f1, bg='red')
l1.pack()

b1 = Button(root, bg='green', fg='white', activebackground='white',
                activeforeground = 'green', text='Map Environment',
                relief=RIDGE, width=16, height=2,command=map)
b1.pack(side='left',padx=60,pady=5)
b2 = Button(root, fg='white',bg='red',activebackground='white',activeforeground='red',
            text='Exit Window',relief=RIDGE,height=2, width=10,command=exitWindow)
b2.pack(side='left',padx=40,pady=5)
b3 = Button(root, bg='blue', fg='white', activebackground='white',
                activeforeground = 'green', text='Get Robot',
                relief=RIDGE, width=16, height=2,command=getrobot)
b3.pack(side='left',padx=40,pady=5)

while True:
    frame = video.read()[1]
    frame = cv2.flip(frame,1)
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
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
        font = cv2.FONT_HERSHEY_SIMPLEX

        if (np.argmax(result) == 1):
            status = "Happy"
            x1, y1, w1, h1 = 0, 0, 175, 75
            cv2.putText(frame, status, (100, 150), font,
                        3, (0, 0, 255), 2, cv2.LINE_4)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255))

        elif (np.argmax(result) == 2):
            status = "Sad"
            cv2.putText(frame, status, (100, 150), font,
                        3, (0, 0, 255), 2, cv2.LINE_4)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255))

        elif (np.argmax(result) == 0):
            status = "Angry"
            x1, y1, w1, h1 = 0, 0, 175, 75
            cv2.putText(frame, status, (100, 150), font,
                        3, (0, 0, 255), 2, cv2.LINE_4)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255))
            mixer.music.load("C:\\Users\\Ryu\\face\\rawr\\MarriedLife.mp3")
            mixer.music.play()

        else:
            status = "none"

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=0) as pose:
        image.flags.writeable = False
        results = pose.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        #landmarks = results.pose_landmarks.landmark
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

        shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                    landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
               landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
        knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]

        angle = calculate_angle(shoulder, hip, knee)

        cv2.putText(frame, str(angle), tuple(np.multiply(hip, [640, 480]).astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

        if angle >= 140:
            stage = "stand"
        if angle < 120 and stage == 'stand':
            stage = "fall"
            engine.say("Fall Detected")
            engine.runAndWait()
            detect += 1

        cv2.rectangle(frame, (0, 0), (100, 70), (195, 144, 98), -1)
        cv2.putText(frame, "Detect", (15, 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(frame, stage, (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (255, 255, 255), 2, cv2.LINE_AA)
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS, mp_drawing.DrawingSpec(color=(
            245, 117, 66), thickness=2, circle_radius=2), mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))

    current_time = datetime.datetime.now().strftime("%H:%M:%S")
    if current_time == alarm_time:
        print("Time to take your medication!")
        engine.say("Hello! It is time to take your medication!")
        engine.runAndWait()
    time.sleep(1)

    frame = ImageTk.PhotoImage(Image.fromarray(image))
    l1['image']=frame
    root.update()

cap.release()

root.mainloop()