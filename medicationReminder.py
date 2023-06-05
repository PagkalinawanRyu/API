import datetime
import time
import pyttsx3

engine = pyttsx3.init()
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[1].id)
engine.setProperty('rate', 125)

alarm_time = "11:30:00"


def set_alarm(alarm_time):
    while True:
        current_time = datetime.datetime.now().strftime("%H:%M:%S")
        if current_time == alarm_time:
            print("Time to take your medication!")
            engine.say("Hello! It is time to take your medication!")
            engine.runAndWait()
            break
        time.sleep(1)


set_alarm(alarm_time)
