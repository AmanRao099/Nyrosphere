#yet to be deployed, testing in progress
import os
import re
import cv2
import numpy as np
import speech_recognition as sr
import pyttsx3
import requests
import google.generativeai as genai

# ESP32 IP for servo control
ESP32_IP = "http://192.168.123.86"

# Configure Gemini
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
gemini_model = genai.GenerativeModel("gemini-1.5-flash")

# TTS Engine Setup
engine = pyttsx3.init()
voices = engine.getProperty('voices')
for voice in voices:
    if "female" in voice.name.lower():
        engine.setProperty('voice', voice.id)
        break

def speak(text):
    print(f"🤖 Assistant: {text}")
    engine.say(text)
    engine.runAndWait()

def listen():
    recognizer = sr.Recognizer()
    try:
        with sr.Microphone() as source:
            print("🎤 Speak now...")
            audio = recognizer.listen(source, phrase_time_limit=5)
    except Exception as e:
        print(f"Microphone error: {e}")
        return None

    try:
        text = recognizer.recognize_google(audio)
        print(f"🧍 You said: {text}")
        return text.lower()
    except sr.UnknownValueError:
        speak("Sorry, I didn't catch that.")
    except sr.RequestError as e:
        print(f"Speech recognition error: {e}")
        speak("Sorry, speech recognition service is down.")
    return None

def ask_gemini(prompt: str) -> str:
    try:
        response = gemini_model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"Gemini error: {e}")
        return "I'm sorry, I couldn't process that request."

def send_to_esp32(angle):
    try:
        response = requests.get(f"{ESP32_IP}/move", params={"angle": angle})
        print(f"✅ Moved to {angle} degrees.")
        speak(f"Moving to {angle} degrees.")
    except:
        speak("❌ Failed to send command to ESP32.")

def capture_image():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        speak("Camera not available.")
        return None
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None

def find_color_angle(frame, color):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    if color == "blue":
        lower, upper = np.array([90, 60, 60]), np.array([130, 255, 255])
        mask = cv2.inRange(hsv, lower, upper)
    elif color == "green":
        lower, upper = np.array([40, 40, 40]), np.array([80, 255, 255])
        mask = cv2.inRange(hsv, lower, upper)
    elif color == "red":
        lower1, upper1 = np.array([0, 70, 50]), np.array([10, 255, 255])
        lower2, upper2 = np.array([170, 70, 50]), np.array([180, 255, 255])
        mask = cv2.bitwise_or(cv2.inRange(hsv, lower1, upper1), cv2.inRange(hsv, lower2, upper2))
    else:
        speak(f"I can't detect the color {color}.")
        return None

    mask = cv2.dilate(cv2.erode(mask, np.ones((5,5), np.uint8), iterations=1), np.ones((5,5), np.uint8), iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        speak(f"I couldn't find any {color} object.")
        return None

    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)
    center_x = x + w // 2
    angle = int((center_x / frame.shape[1]) * 180)
    print(f"🎯 Detected {color} at X={center_x}, Angle={angle}")
    return angle

def extract_angle(command):
    match = re.search(r'\b(\d{1,3})\b', command)
    if match:
        angle = int(match.group(1))
        if 0 <= angle <= 180:
            return angle
    return None

def run_assistant():
    speak("Hello! I am your assistant. Ask me anything.")
    while True:
        command = listen()
        if not command:
            continue

        if command in ["exit", "quit", "stop"]:
            speak("Goodbye!")
            break

        # 1. Direct angle control
        angle = extract_angle(command)
        if angle is not None:
            send_to_esp32(angle)
            continue

        # 2. Color-based command
        if "point to" in command or "look at" in command:
            color = command.split()[-1]
            frame = capture_image()
            if frame is not None:
                speak(f"Scanning for {color}...")
                angle = find_color_angle(frame, color)
                if angle is not None:
                    send_to_esp32(angle)
            continue

        # 3. General AI answer via Gemini
        response = ask_gemini(command)
        speak(response)

if __name__ == "__main__":
    if not os.getenv("GOOGLE_API_KEY"):
        print("❌ Please set the GOOGLE_API_KEY environment variable.")
        speak("Gemini key is missing. Please configure it.")
        exit(1)
    run_assistant()
