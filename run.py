"""The main script to run people recognition on camera video."""
from pathlib import Path

import cv2
from face_recognizer import FaceRecognizer
from mediapipe.python.solutions.face_detection import FaceDetection
from mediapipe.python.solutions.hands import Hands
from people_recognizer import PeopleRecognizer

if __name__ == "__main__":

    labeled_face_dir = Path(r"C:\Projects\FHL\face_recognition\labeled_faces\jfif")
    face_recognizer = FaceRecognizer().encode_face_images_in_dir(labeled_face_dir, ext="jfif")
    face_detector = FaceDetection(model_selection=0, min_detection_confidence=0.5)
    hand_detector = Hands(model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5)

    people_recognizer = PeopleRecognizer(face_detector, face_recognizer, hand_detector)
    video_capturer = cv2.VideoCapture(1)
    people_recognizer.recognize_people_in_video(video_capturer)
