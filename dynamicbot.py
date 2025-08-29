# dynamicbot_fixed.py
import ollama
import pyttsx3
import speech_recognition as sr
import pvporcupine
import pyaudio
import struct
import threading
import time
import datetime
import os
from dotenv import load_dotenv


# ====== Real time data time day ======
def get_current_date():
    today = datetime.date.today()
    return f"Today's date is {today.strftime('%B %d, %Y')}."

def get_current_day():
    today = datetime.date.today()
    return f"Today is {today.strftime('%A')}."

def get_current_time():
    now = datetime.datetime.now()
    return f"The current time is {now.strftime('%I:%M %p')}."



# ====== Wake Word Detector ======
class WakeWordDetector:
    def __init__(self, access_key, keyword="jarvis"):
        self.porcupine = pvporcupine.create(
            access_key=access_key,
            keywords=[keyword]
        )
        self.pa = pyaudio.PyAudio()
        self.audio_stream = self.pa.open(
            rate=self.porcupine.sample_rate,
            channels=1,
            format=pyaudio.paInt16,
            input=True,
            frames_per_buffer=self.porcupine.frame_length
        )
        self.keyword = keyword

    def listen(self):
        print("") #listen for wake word
        while True:
            pcm = self.audio_stream.read(self.porcupine.frame_length, exception_on_overflow=False)
            pcm = struct.unpack_from("h" * self.porcupine.frame_length, pcm)
            if self.porcupine.process(pcm) >= 0:
                print(f" : {self.keyword} booting up...")
                return True

    def close(self):
        self.audio_stream.close()
        self.pa.terminate()
        self.porcupine.delete()




# ====== Voice Bot ======
engine = pyttsx3.init()
engine.setProperty("rate", 180)
engine.setProperty("volume", 1)

recognizer = sr.Recognizer()
mic = sr.Microphone()

def speak(text):
    print("🤖 Bot:", text)
    engine.say(text)
    engine.runAndWait()

def listen():
    with mic as source:
        print(" Listening...")
        recognizer.adjust_for_ambient_noise(source, duration=0.3)
        try:
            audio = recognizer.listen(source, timeout=3, phrase_time_limit=3)
        except sr.WaitTimeoutError:
            return None
    try:
        return recognizer.recognize_google(audio).lower()
    except sr.UnknownValueError:
        return None
    except sr.RequestError:
        speak("Speech recognition service is unavailable.")
        return None

def chat_with_bot(prompt):
    response_text = ""
    stream = ollama.chat(
        model="llama3.2",
        messages=[{"role": "user", "content": prompt}],
        stream=True,
    )
    for chunk in stream:
        if "message" in chunk and "content" in chunk["message"]:
            response_text += chunk["message"]["content"]
    return response_text.strip()

def summarize_response(response):
    sentences = response.replace("\n", " ").split(". ")
    summary_points = sentences[:3]
    return "\n".join(f"- {s.strip()}" for s in summary_points if s.strip())

def bot_main():
    
    speak("Hello! I'm your assistant. How may i help you ")
    while True:
        user_input = listen()
        if not user_input:
            speak("Sorry, I didn't catch that.")
            continue

        if any(cmd in user_input for cmd in ["exit", "quit", "stop"]):
            speak("Goodbye!")
            break
        
         # ====== Real-time responses ======
        if "date" in user_input:
            speak(get_current_date())
            continue
        if "day" in user_input:
            speak(get_current_day())
            continue
        if "time" in user_input:
            speak(get_current_time())
            continue

        bot_response = chat_with_bot(user_input)

        if any(word in user_input for word in ["detail", "detailed", "full", "explain", "tell me more"]):
            speak(bot_response)
        else:
            speak(summarize_response(bot_response))
             
             
        

# ====== Main ======

if __name__ == "__main__":
    load_dotenv()
    ACCESS_KEY = os.getenv("PICOVOICE_ACCESS_KEY")
    wake_detector = WakeWordDetector(ACCESS_KEY)

    try:
        while True:
            if wake_detector.listen():  # Wait for wake word
                bot_main()
                print("") #🔄 Back to listening for wake word...

    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        wake_detector.close()
if __name__ == "__main__":
    bot_main()
    input("Press Enter to exit...")  # Keeps window open
