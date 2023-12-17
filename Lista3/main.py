import cv2
import face_recognition
import os

# Create arrays of known face encodings and corresponding names
known_face_encodings = []
known_face_names = []

process_current_frame = True

for image in os.listdir('faces'):
    face_image = face_recognition.load_image_file(f'faces/{image}')
    print(image)
    face_encoding = face_recognition.face_encodings(face_image)[0]

    known_face_encodings.append(face_encoding)
    known_face_names.append(image)

print(known_face_names)

# Open a video capture
video_capture = cv2.VideoCapture(0)

while True:
    # Capture each frame from the video feed
    # Find all face locations and face encodings in the current frame
    if process_current_frame:
        ret, frame = video_capture.read()
        face_locations = face_recognition.face_locations(frame)
        face_encodings = face_recognition.face_encodings(frame, face_locations)

    # Loop through each face found in the frame
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Check if the face matches any known faces
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

        name = "Unknown"

        # If a match is found, use the name of the known person
        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]

        # Draw a rectangle around the face and display the name
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

    # Display the resulting image
    cv2.imshow("Video", frame)

    process_current_frame = not process_current_frame

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
video_capture.release()
cv2.destroyAllWindows()

images_folder = "./images"

# Loop through each image in the folder
for filename in os.listdir(images_folder):
    if filename.endswith((".jpg", ".jpeg", ".png")):
        # Load the image to recognize
        image_to_recognize = face_recognition.load_image_file(os.path.join(images_folder, filename))
        face_locations = face_recognition.face_locations(image_to_recognize)
        face_encodings = face_recognition.face_encodings(image_to_recognize, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # Check if the face matches any known faces
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

            name = "Unknown"

            # If a match is found, use the name of the known person
            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]

            # Draw a rectangle around the face and display the name
            cv2.rectangle(image_to_recognize, (left, top), (right, bottom), (0, 0, 255), 2)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(image_to_recognize, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

        image_to_recognize_rgb = cv2.cvtColor(image_to_recognize, cv2.COLOR_BGR2RGB)

        # Display the resulting image with recognized faces
        cv2.imshow("Image Recognition", image_to_recognize_rgb)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
