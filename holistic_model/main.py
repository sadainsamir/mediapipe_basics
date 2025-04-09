import cv2
import mediapipe as mp

# Initialize Holistic model
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

# Load image
image_path = "/home/sadainsamir/Downloads/pose1.jpg"
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Process with Holistic
with mp_holistic.Holistic(static_image_mode=True) as holistic:
    results = holistic.process(image_rgb)

    # FACE LANDMARKS
    if results.face_landmarks:
        print("\nFace Landmarks:")
        for i, landmark in enumerate(results.face_landmarks.landmark):
            print(f"  {i}: x={landmark.x}, y={landmark.y}, z={landmark.z}")

    # POSE LANDMARKS
    if results.pose_landmarks:
        print("\nPose Landmarks:")
        for i, landmark in enumerate(results.pose_landmarks.landmark):
            print(f"  {i}: x={landmark.x}, y={landmark.y}, z={landmark.z}")

    # WORLD POSE LANDMARKS
    if results.pose_world_landmarks:
        print("\nPose World Landmarks:")
        for i, landmark in enumerate(results.pose_world_landmarks.landmark):
            print(f"  {i}: x={landmark.x}, y={landmark.y}, z={landmark.z}")

    # LEFT HAND
    if results.left_hand_landmarks:
        print("\nLeft Hand Landmarks:")
        for i, landmark in enumerate(results.left_hand_landmarks.landmark):
            print(f"  {i}: x={landmark.x}, y={landmark.y}, z={landmark.z}")

    # RIGHT HAND
    if results.right_hand_landmarks:
        print("\nRight Hand Landmarks:")
        for i, landmark in enumerate(results.right_hand_landmarks.landmark):
            print(f"  {i}: x={landmark.x}, y={landmark.y}, z={landmark.z}")
