import cv2
import face_recognition
import numpy as np
import os

# --- Constants ---
ENCODINGS_FILE = "known_faces.npy"
NAMES_FILE = "known_names.txt"
# Lowering the tolerance makes face comparison stricter. 0.6 is the default,
# but a value like 0.5 or 0.45 can be more accurate for distinct faces.
RECOGNITION_TOLERANCE = 0.3

# --- Helper Functions for Data Persistence ---

def load_known_faces(encodings_path, names_path):
    """Loads face encodings and names from files."""
    known_face_encodings = []
    known_face_names = []
    if os.path.exists(encodings_path) and os.path.exists(names_path):
        known_face_encodings = list(np.load(encodings_path))
        with open(names_path, "r") as f:
            known_face_names = [line.strip() for line in f.readlines()]
        print(f"Loaded {len(known_face_names)} known faces and names.")
    return known_face_encodings, known_face_names

def save_known_faces(encodings, names, encodings_path, names_path):
    """Saves face encodings and names to files."""
    np.save(encodings_path, np.array(encodings))
    with open(names_path, "w") as f:
        for name in names:
            f.write(f"{name}\n")
    print(f"Saved {len(names)} known faces and names.")


# --- Main Function ---
def main():
    """
    Main function to run the face recognition and learning application.
    """
    # --- Initialization ---
    
    # Load any previously saved faces
    known_face_encodings, known_face_names = load_known_faces(ENCODINGS_FILE, NAMES_FILE)

    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    # Variables for processing frames and storing results
    face_locations = []
    face_encodings = []
    face_names = []
    process_this_frame = True
    
    print("Starting webcam feed. Press 'q' to quit.")
    print("When an 'Unknown' face is detected, press 's' to save it.")

    # --- Main Loop ---
    while True:
        # Read a frame from the webcam
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        # --- Frame Processing for Performance ---
        # Only process every other frame of video to save time
        if process_this_frame:
            # For performance, we'll process a smaller version of the frame.
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            
            # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

            # --- Face Detection and Recognition ---
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            face_names = []
            for face_encoding in face_encodings:
                name = "Unknown"
                # Calculate the distance between the new face and all known faces
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                
                # If there are any known faces, find the best match
                if len(face_distances) > 0:
                    best_match_index = np.argmin(face_distances)
                    # If the best match distance is within our tolerance, we have a match
                    if face_distances[best_match_index] < RECOGNITION_TOLERANCE:
                        name = known_face_names[best_match_index]
                
                face_names.append(name)

        # Toggle the flag to skip the next frame
        process_this_frame = not process_this_frame

        # --- Display Results ---
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Scale back up face locations
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            # Draw a box around the face
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255) # Green for known, Red for unknown
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        # Display the resulting image
        cv2.imshow('Face Recognition', frame)

        # --- User Interaction ---
        key = cv2.waitKey(1) & 0xFF
        
        # Quit the program if 'q' is pressed
        if key == ord('q'):
            break
        
        # Save a new face if 's' is pressed
        if key == ord('s'):
            if face_names.count("Unknown") == 1 and len(face_encodings) == 1:
                new_face_encoding = face_encodings[0]
                
                name = input("Enter the name for the new face and press Enter: ")
                
                if name:
                    known_face_encodings.append(new_face_encoding)
                    known_face_names.append(name)
                    # Save the updated lists to files
                    save_known_faces(known_face_encodings, known_face_names, ENCODINGS_FILE, NAMES_FILE)
                else:
                    print("Invalid name. Face not saved.")
            elif len(face_encodings) == 0:
                print("No face detected. Cannot save.")
            else:
                print("Multiple faces detected or face is already known. Please ensure only one 'Unknown' face is visible.")

    # --- Cleanup ---
    cap.release()
    cv2.destroyAllWindows()
    print("Webcam feed stopped.")

if __name__ == '__main__':
    main()
