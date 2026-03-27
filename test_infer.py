import urllib.request
import cv2
import mediapipe as mp
import numpy as np

# Download a sample face image
url = "https://raw.githubusercontent.com/google/mediapipe/master/mediapipe/objc/mediapipe_framework_ios.cc" # just need any valid file to check network... wait no, need an image.
url = "https://avatars.githubusercontent.com/u/100"
urllib.request.urlretrieve(url, "test_face.jpg")

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="models/face_landmarker.task"),
    running_mode=RunningMode.IMAGE,
    num_faces=1,
)

with FaceLandmarker.create_from_options(options) as landmarker:
    img = cv2.imread("test_face.jpg")
    if img is None:
        # Create a dummy image with a white circle (might not detect a face, but let's try)
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.circle(img, (320, 240), 100, (255, 255, 255), -1)
        
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    try:
        result = landmarker.detect(mp_image)
        if result.face_landmarks:
            print("Face detected!")
            lms = result.face_landmarks[0]
            for i in [468, 469, 470, 471, 472]:
                print(f"Landmark {i} visibility: {lms[i].visibility}, presence: {lms[i].presence}")
        else:
            print("No face detected in test image.")
    except Exception as e:
        print("Error:", e)
