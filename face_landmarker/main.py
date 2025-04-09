import cv2
import mediapipe as mp
import numpy as np 
from mediapipe.tasks import python # api structure
from mediapipe.tasks.python import vision # vision specific models like FaceLandmarker

model = '/home/sadainsamir/MediaPipeProject/face_landmarker/model/face_landmarker.task'

base_options = python.BaseOptions(model_asset_path=model)

options = vision.FaceLandmarkerOptions(
	base_options = base_options,
	output_face_blendshapes = True, # includes facial expression info
	# output_facial_transformation_matrix = True, # include 3d head pose matrix
	num_faces=1 # detects only 1 face
)

image_path = '/home/sadainsamir/MediaPipeProject/face_landmarker/input_images/01_img.jpg'
image_bgr = cv2.imread(image_path)

image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

mp_image = mp.Image(image_format = mp.ImageFormat.SRGB, data=image_rgb)

with vision.FaceLandmarker.create_from_options(options) as detector:
	detection_result = detector.detect(mp_image)
	for blendshape in detection_result.face_blendshapes[0]:
	    print(f"{blendshape.category_name} - {blendshape.score}")


	if detection_result.face_landmarks:
		for idx, face_landmarks in enumerate(detection_result.face_landmarks):
			print(f"\nFace {idx+1} Landmarks :")
			for i, landmark in enumerate(face_landmarks):
				print(landmark)

	if detection_result.facial_transformation_matrixes:
		print('\nFacial Transformation Matrix:')
		print(detection_result.facial_transformation_matrixes[0])

