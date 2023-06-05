import speech_recognition as sr
import pyttsx3


def navigation():

    engine = pyttsx3.init()
    voices = engine.getProperty('voices')
    engine.setProperty('voice', voices[1].id)
    engine.setProperty('rate', 125)
    r = sr.Recognizer()

    with sr.Microphone() as source:

        while True:

            print("Ready for Speech...")
            audio_data = r.listen(source)
            text = r.recognize_google(audio_data)

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


navigation()
