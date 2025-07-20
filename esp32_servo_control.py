import speech_recognition as sr
import pyttsx3
import requests
import re
import cv2
import numpy as np
from transformers import pipeline

# Replace with your ESP32 IP address
ESP32_IP = "http://192.168.123.86"

# Load local QA model (Free, no API key needed)
qa_pipeline = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")

# Initialize TTS engine with female voice
engine = pyttsx3.init()
voices = engine.getProperty('voices')
for voice in voices:
    if "female" in voice.name.lower():
        engine.setProperty('voice', voice.id)
        break

def speak(text):
    print("Assistant:", text)
    engine.say(text)
    engine.runAndWait()

def recognize_command():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        speak("Say a command.")
        print("🎤 Listening...")
        audio = recognizer.listen(source)

    try:
        command = recognizer.recognize_google(audio).lower()
        print("🔊 You said:", command)
        return command
    except sr.UnknownValueError:
        speak("Sorry, I didn't catch that.")
    except sr.RequestError:
        speak("Speech recognition failed.")
    return None

def extract_angle(command):
    match = re.search(r'\b(\d{1,3})\b', command)
    if match:
        angle = int(match.group(1))
        if 0 <= angle <= 180:
            return angle
    return None

def send_to_esp32(angle):
    try:
        response = requests.get(f"{ESP32_IP}/move", params={"angle": angle})
        print("✅ Moved to", angle, "degrees.")
        speak(f"Moving to {angle} degrees.")
    except:
        speak("Failed to send command to ESP32.")

def capture_image():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        speak("Camera not available.")
        return None
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None

def find_color_angle(frame, color):
    """
    Detect the largest blob of the specified color in the frame.
    Return the servo angle (0-180) corresponding to the horizontal position of the color.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    if color == "blue":
        lower = np.array([90, 60, 60])
        upper = np.array([130, 255, 255])
    elif color == "red":
        lower1 = np.array([0, 70, 50])
        upper1 = np.array([10, 255, 255])
        lower2 = np.array([170, 70, 50])
        upper2 = np.array([180, 255, 255])
        mask1 = cv2.inRange(hsv, lower1, upper1)
        mask2 = cv2.inRange(hsv, lower2, upper2)
        mask = cv2.bitwise_or(mask1, mask2)
    elif color == "green":
        lower = np.array([40, 40, 40])
        upper = np.array([80, 255, 255])
    else:
        speak(f"Sorry, I cannot detect the color {color}.")
        return None

    if color != "red":
        mask = cv2.inRange(hsv, lower, upper)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        speak(f"I couldn't find any {color} object.")
        return None

    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    center_x = x + w // 2
    img_width = frame.shape[1]
    angle = int((center_x / img_width) * 180)

    print(f"Detected {color} object at pixel {center_x}, mapped to angle {angle}")
    return angle

def analyze_image_options(frame, question):
    """Slice image into 3 vertical sections and choose best matching option"""
    h, w = frame.shape[:2]
    slice_width = w // 3
    options = [frame[:, i*slice_width:(i+1)*slice_width] for i in range(3)]

    # Dummy labels for demonstration - replace with real image classification for better accuracy
    labels = ["apple", "banana", "car", "cat", "tree"]
    detected_labels = []

    for i, opt in enumerate(options):
        hsv = cv2.cvtColor(opt, cv2.COLOR_BGR2HSV)
        mean = cv2.mean(hsv)
        # Dummy heuristic: just pick a label cyclically for now
        detected_labels.append(labels[i % len(labels)])

    best_score = -1
    best_index = None
    for idx, label in enumerate(detected_labels):
        result = qa_pipeline(question=question, context=label)
        if result['score'] > best_score:
            best_score = result['score']
            best_index = idx

    print("Detected options:", detected_labels)
    angle = int((best_index + 0.5) * 60)  # roughly 0°, 60°, or 120° servo angle
    return angle, detected_labels[best_index]

def main():
    while True:
        command = recognize_command()
        if not command:
            continue

        angle = extract_angle(command)
        if angle is not None:
            send_to_esp32(angle)
            continue

        if "point to" in command or "look at" in command:
            color = command.split()[-1]
            frame = capture_image()
            if frame is None:
                continue
            speak(f"Finding {color} object...")
            angle = find_color_angle(frame, color)
            if angle is not None:
                send_to_esp32(angle)
            continue

        if "which" in command or "choose" in command or "identify" in command:
            speak("Capturing image to analyze options.")
            frame = capture_image()
            if frame is None:
                continue
            angle, label = analyze_image_options(frame, command)
            speak(f"I think the correct answer is {label}")
            send_to_esp32(angle)
            continue

        # Fallback to text-based QA
        context = "Narendra Modi is the current Prime Minister of India. Apple is a fruit. Blue is a color."
        try:
            result = qa_pipeline(question=command, context=context)
            speak(result['answer'])
        except Exception as e:
            print("AI Error:", e)
            speak("Sorry, I couldn't answer that.")

if __name__ == "__main__":
    main()
