"""
Real-Time Sign Language Recognition - Simple Version
Basic prediction display without confidence bar
"""
import cv2
import mediapipe as mp
import numpy as np
import joblib

# Load the trained model and labels
model = joblib.load("model.pkl")
label_encoder = joblib.load("labels.pkl")

# MediaPipe Hands (using Tasks API)
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


def create_hand_landmarker():
    """Create and configure the hand landmarker."""
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.7,
        min_hand_presence_confidence=0.7,
        min_tracking_confidence=0.7,
    )
    return HandLandmarker.create_from_options(options)


# Create the landmarker
landmarker = create_hand_landmarker()

# Camera setup
cap = cv2.VideoCapture(0)

print("Starting real-time prediction... Press Q to quit")

frame_timestamp_ms = 0

while True:
    # Get frame from camera
    ret, frame = cap.read()
    if not ret:
        break

    # Mirror the frame
    frame = cv2.flip(frame, 1)
    frame_timestamp_ms += 1

    # Convert to RGB for MediaPipe
    rgb_data = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb_frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_data)

    result = landmarker.detect_for_video(rgb_frame, frame_timestamp_ms)

    prediction_text = "No hand detected"

    if result.hand_landmarks:
        for hand_landmarks in result.hand_landmarks:
            # Draw landmarks
            for lm in hand_landmarks:
                x, y = int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

            # Draw connections
            connections = [
                (0, 1), (1, 2), (2, 3), (3, 4),  # thumb
                (0, 5), (5, 6), (6, 7), (7, 8),  # index
                (0, 9), (9, 10), (10, 11), (11, 12),  # middle
                (0, 13), (13, 14), (14, 15), (15, 16),  # ring
                (0, 17), (17, 18), (18, 19), (19, 20),  # pinky
                (5, 9), (9, 13), (13, 17)  # palm
            ]
            for start, end in connections:
                start_lm = hand_landmarks[start]
                end_lm = hand_landmarks[end]
                x1, y1 = int(start_lm.x * frame.shape[1]), int(start_lm.y * frame.shape[0])
                x2, y2 = int(end_lm.x * frame.shape[1]), int(end_lm.y * frame.shape[0])
                cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

            # Extract landmarks
            row = []
            for lm in hand_landmarks:
                row.extend([lm.x, lm.y, lm.z])

            # Convert to numpy
            X = np.array(row).reshape(1, -1)

            # Predict
            pred = model.predict(X)[0]
            label = label_encoder.inverse_transform([pred])[0]

            prediction_text = f"Prediction: {label}"

    # Display text
    cv2.putText(
        frame,
        prediction_text,
        (10, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("Real-Time Sign Language Recognition", frame)

    # Press Q to quit
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
