# mediapipe_basics

This repository contains basic implementations of various [MediaPipe](https://mediapipe.dev/) models using Python. The goal of this project was to understand how MediaPipe's pre-trained models work and how to use them on static images for tasks such as:

- Face Detection
- Face Landmark Detection (with blendshapes)
- Hand Tracking
- Holistic Model (full body pose + face + hands)

---

## 📁 Project Structure

```bash
mediapipe_basics/
│
├── face_detection/
│   ├── main.py
│   ├── input_images/         # Not tracked on GitHub
│   ├── output_images/        # Not tracked on GitHub
│   ├── output_data/          # Not tracked on GitHub
│
├── face_landmarker/
│   ├── main.py
│   ├── model/
│   │   └── face_landmarker.task
│   ├── input_images/         # Not tracked
│   └── output_face_landmarker/
│
├── hand_tracking/
│   ├── main.py
│   ├── model/
│   │   └── hand_landmarker.task
│   └── images/               # Not tracked
│
├── holistic_model/
│   ├── main.py
│   ├── model/
│   └── images_pose/          # Not tracked
│
├── envs/                     # Virtual environments (ignored)
├── __archive__/              # Old versions (ignored)
├── .gitignore
└── README.md
