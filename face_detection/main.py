import cv2
import os
import csv
from mediapipe import solutions
from mediapipe.framework.formats import detection_pb2

# Create face detector object from MediaPipe
face_detection = solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

# Path setup
input_folder = "/home/sadainsamir/MediaPipeProject/face_detection/input_images"
output_img_folder = "/home/sadainsamir/MediaPipeProject/face_detection/output_images"
output_csv_folder = "/home/sadainsamir/MediaPipeProject/face_detection/output_data"

# Get all images from input folder
image_files = [f for f in os.listdir(input_folder) if f.endswith(('.jpg', '.png'))]

# Loop over each image
for img_file in image_files:
    img_path = os.path.join(input_folder, img_file)
    image = cv2.imread(img_path)

    # Convert BGR to RGB because MediaPipe expects RGB
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Run detection
    results = face_detection.process(rgb_image)

    # Prepare output structures
    output_image = image.copy()
    output_data = []

    if results.detections:
        for i, detection in enumerate(results.detections):
            # Extract bounding box
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = image.shape
            x = int(bboxC.xmin * iw)
            y = int(bboxC.ymin * ih)
            w = int(bboxC.width * iw)
            h = int(bboxC.height * ih)

            # Draw rectangle on image
            cv2.rectangle(output_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Extract detection score
            score = detection.score[0]

            # Save to output list
            output_data.append({
                "Face #": i + 1,
                "Detection Score": round(score, 3),
                "X": x,
                "Y": y,
                "Width": w,
                "Height": h
            })
    else:
        print(f"No face detected in {img_file}")

    # Save annotated image
    annotated_path = os.path.join(output_img_folder, f"annotated_{img_file}")
    cv2.imwrite(annotated_path, output_image)

    # Save CSV
    csv_file_path = os.path.join(output_csv_folder, f"{os.path.splitext(img_file)[0]}.csv")
    with open(csv_file_path, mode='w', newline='') as csv_file:
        fieldnames = ["Face #", "Detection Score", "X", "Y", "Width", "Height"]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in output_data:
            writer.writerow(row)

    print(f"✅ Processed {img_file}: {len(output_data)} face(s) detected")
