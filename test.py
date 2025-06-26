import os
import cv2
import face_recognition
import streamlit as st
from PIL import Image
import numpy as np

st.set_page_config(layout="wide")
st.title("🎥 Real-Time Face Recognition from Video")

# --- Load Known Faces ---
@st.cache_data(show_spinner=True)
def load_known_faces(known_dir):
    encodings = []
    names = []
    ages = []

    for age_group in os.listdir(known_dir):
        age_path = os.path.join(known_dir, age_group)
        if not os.path.isdir(age_path):
            continue

        for file in os.listdir(age_path):
            path = os.path.join(age_path, file)
            try:
                image = face_recognition.load_image_file(path)
                faces = face_recognition.face_encodings(image)
                if faces:
                    encodings.append(faces[0])
                    names.append(os.path.splitext(file)[0])
                    ages.append(age_group)
            except Exception as e:
                print(f"Error loading {path}: {e}")
    return encodings, names, ages

# Load faces from your directory
KNOWN_DIR = "/workspaces/Face_Recognition-New/Main"
known_face_encodings, known_face_names, known_face_age_groups = load_known_faces(KNOWN_DIR)

# --- Upload or Use Default Video ---
video_path = st.file_uploader("Upload a video", type=["mp4, jpg, jpeg, png"], label_visibility="collapsed")
if not video_path:
    video_path = "/workspaces/Face_Recognition-New/WhatsApp Video 2024-02-07 at 19.20.24_faf9da60.mp4"

cap = cv2.VideoCapture(video_path)

frame_placeholder = st.empty()

frame_count = 0
max_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# --- Stream Video with Face Recognition ---
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        st.info("📽️ End of video.")
        break

    frame_count += 1

    # Resize for faster processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small = small_frame[:, :, ::-1]

    # Detect faces
    face_locations = face_recognition.face_locations(rgb_small)
    face_encodings = face_recognition.face_encodings(rgb_small, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"
        age_group = ""

        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        if len(face_distances) > 0:
            best_match = np.argmin(face_distances)
            if matches[best_match]:
                name = known_face_names[best_match]
                age_group = known_face_age_groups[best_match]

        # Scale up for display
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        label = f"{name} ({age_group})" if age_group else name
        cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Show annotated frame in Streamlit
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)

    # Optional: Add a short delay (adjust FPS)
    # time.sleep(1 / 30)

cap.release()
