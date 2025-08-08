import cv2
import numpy as np
import mediapipe as mp
import time
from collections import deque

# --- Main Function ---
def main():
    """
    Main function to run the hand tracking and drawing application.
    """
    # --- Initialization ---
    
    # Initialize MediaPipe Hands
    mp_hands = mp.solutions.hands
    # The 'with' block automatically handles the closing of the hands resources.
    # min_detection_confidence: More confident hand detections are considered.
    # min_tracking_confidence: More confident tracking reduces jitter but might lose the hand if it moves fast.
    hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
    mp_drawing = mp.solutions.drawing_utils

    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Create a deque (double-ended queue) to store the points of the path.
    # A deque has a maximum length, so older points will be automatically removed.
    # This prevents the line from growing infinitely long.
    points = deque(maxlen=1024)

    # Get webcam frame dimensions to create a canvas
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture initial frame.")
        cap.release()
        return
    
    # Create a black image (canvas) with the same dimensions as the webcam frame
    canvas = np.zeros_like(frame)
    
    print("Starting webcam feed. Press 'q' to quit.")
    print("Move your index finger to draw on the black canvas.")

    # --- Main Loop ---
    while True:
        # Read a frame from the webcam
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        # Flip the frame horizontally for a more intuitive "mirror" view
        frame = cv2.flip(frame, 1)
        
        # Convert the BGR frame to RGB, as MediaPipe expects RGB input
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame to find hand landmarks
        results = hands.process(rgb_frame)

        # Get frame dimensions
        height, width, _ = frame.shape

        # --- Hand Landmark Processing ---
        
        # Check if any hands were detected
        if results.multi_hand_landmarks:
            # Iterate through all detected hands
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw the hand skeleton on the main video frame
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Get the coordinates of the index finger tip (landmark #8)
                index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                
                # Convert normalized coordinates (0.0 to 1.0) to pixel coordinates
                center_x, center_y = int(index_finger_tip.x * width), int(index_finger_tip.y * height)
                
                # Add the current point to our deque
                points.appendleft((center_x, center_y))

        # --- Drawing on the Canvas ---
        
        # Iterate through the points in the deque and draw lines
        for i in range(1, len(points)):
            # If either of the points is None, skip this iteration
            if points[i - 1] is None or points[i] is None:
                continue
            
            # Draw a line between the current point and the previous point
            # This creates the continuous drawing effect
            cv2.line(canvas, points[i - 1], points[i], (0, 255, 0), 2)

        # --- Display Windows ---
        
        # Display the main webcam feed with hand tracking
        cv2.imshow('Hand Tracking', frame)
        # Display the separate drawing canvas
        cv2.imshow('Drawing Canvas', canvas)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # --- Cleanup ---
    hands.close()
    cap.release()
    cv2.destroyAllWindows()
    print("Webcam feed stopped.")

if __name__ == '__main__':
    main()
