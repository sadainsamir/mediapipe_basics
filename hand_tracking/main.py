import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

model_path = "/home/sadainsamir/MediaPipeProject/hand_tracking/model/hand_landmarker.task"

base_options = python.BaseOptions(model_asset_path=model_path)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2,
    running_mode=vision.RunningMode.IMAGE
)
# Create the detector object
detector = vision.HandLandmarker.create_from_options(options)

# Load image using OpenCV
image_path = "/home/sadainsamir/Downloads/testPics/hand3.jpg"  # ← Replace with actual image path
image_bgr = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

# Wrap it in a MediaPipe Image format
mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

# Perform detection
detection_result = detector.detect(mp_image)

# Check if any hand was detected
if detection_result.hand_landmarks:
    for hand_index, hand_landmarks in enumerate(detection_result.hand_landmarks):
        print(f"\nHand {hand_index + 1} Landmarks:")
        for idx, landmark in enumerate(hand_landmarks):
            print(f"  Landmark {idx}: x = {landmark.x}, y = {landmark.y}, z = {landmark.z}")
else:
    print("No hands detected.")




