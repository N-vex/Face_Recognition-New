import os
import face_recognition
import cv2

# Load known faces
known_face_encodings = []
known_face_names = []
known_face_age_groups = []
known_dir = "/workspaces/Face_Recognition-New/Main"

for age_group in os.listdir(known_dir):
    age_group_path = os.path.join(known_dir, age_group)
    if not os.path.isdir(age_group_path):
        continue

    for filename in os.listdir(age_group_path):
        file_path = os.path.join(age_group_path, filename)
        try:
            image = face_recognition.load_image_file(file_path)
            encodings = face_recognition.face_encodings(image)
            if encodings:
                known_face_encodings.append(encodings[0])
                known_face_names.append(os.path.splitext(filename)[0])
                known_face_age_groups.append(age_group)
            else:
                print(f"[!] No face found: {file_path}")
        except Exception as e:
            print(f"[ERROR] Failed to process {file_path}: {e}")

# Test with one image
test_image_path = "/workspaces/Face_Recognition-New/Main/21-30 (10).jpg"
test_image = face_recognition.load_image_file(test_image_path)
face_locations = face_recognition.face_locations(test_image)
face_encodings = face_recognition.face_encodings(test_image, face_locations)

for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
    matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
    name = "Unknown"

    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
    best_match_index = face_distances.argmin()
    if matches[best_match_index]:
        name = known_face_names[best_match_index]

    cv2.rectangle(test_image, (left, top), (right, bottom), (0, 255, 0), 2)
    cv2.putText(test_image, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

# Save test image result
rgb_img = cv2.cvtColor(test_image, cv2.COLOR_RGB2BGR)
cv2.imwrite("output_test_image.jpg", rgb_img)
print("Saved result as output_test_image.jpg")

# Process video file
video_path = "/workspaces/Face_Recognition-New/WhatsApp Video 2024-02-07 at 19.20.24_faf9da60.mp4"
video_capture = cv2.VideoCapture(video_path)

if not video_capture.isOpened():
    print(f"❌ Error: Could not open video file: {video_path}")
    exit()

frame_count = 0
while True:
    ret, frame = video_capture.read()
    if not ret:
        print("✅ End of video or failed to grab frame.")
        break

    frame_count += 1
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = small_frame[:, :, ::-1]

    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = face_distances.argmin()
        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        # Scale back up
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

        # Save the first successful frame with a match
        cv2.imwrite("output_video_match.jpg", frame)
        print(f"🎯 Match found and saved in frame {frame_count} -> output_video_match.jpg")
        break

video_capture.release()
print("✅ Video processing complete.")
"""  """