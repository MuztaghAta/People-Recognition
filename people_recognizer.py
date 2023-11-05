from typing import List, Optional, Tuple

import cv2
import face_recognition as fr
import numpy as np
from face_recognizer import FaceRecognizer
from mediapipe.framework.formats.location_data_pb2 import LocationData
from mediapipe.python.solutions import drawing_styles, drawing_utils
from mediapipe.python.solutions.face_detection import FaceDetection
from mediapipe.python.solutions.hands import HAND_CONNECTIONS, Hands

FaceLocation = Tuple[int, int, int, int]
NOT_A_FACE: FaceLocation = (-1, -1, -1, -1)


class PeopleRecognizer:
    """Recognize people in camera video.

    Right now, it's able to detect faces and hands, and recognize faces.

    TODO:
    - recognize to whom the detected hands belongs
    - record history (e.g. who were recognized at what time, who raised hands at what time)
    - enable voice detection and recognition and connect to face/hands/whatever human objects
    - Most importantly, make the recognition faster.

    Ref:
    - mediapipe:
        - (doc)[https://google.github.io/mediapipe/getting_started/python.html]
        - (code)[https://github.com/google/mediapipe/tree/master/mediapipe]
        - (homepage)[https://mediapipe.dev/]
    """

    def __init__(
        self,
        face_detector: Optional[FaceDetection] = None,
        face_recognizer: Optional[FaceRecognizer] = None,
        hand_detector: Optional[Hands] = None,
        interval: int = 1,
    ):
        self.face_detector = face_detector
        self.face_recognizer = face_recognizer
        self.hand_detector = hand_detector
        self.interval = interval

    def recognize_people_in_video(self, video_capturer: cv2.VideoCapture):
        n = 0
        while video_capturer.isOpened():
            n += 1
            success, image = video_capturer.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue  # use 'break' instead of 'continue' if loading a video
            image = cv2.flip(image, 1)
            if n % self.interval == 0:
                image = self.recognize_people_in_image(
                    image,
                    self.face_detector,
                    self.face_recognizer,
                    self.hand_detector,
                )
            cv2.imshow("Video", image)
            ms_per_frame = 5
            quit_key = "q"
            if cv2.waitKey(ms_per_frame) & 0xFF == ord(quit_key):
                break
        video_capturer.release()  # ISSUE: camera LED is still on afterwards
        cv2.destroyAllWindows()

    def recognize_people_in_image(
        self,
        image: np.ndarray,
        face_detector: Optional[FaceDetection] = None,
        face_recognizer: Optional[FaceRecognizer] = None,
        hand_detector: Optional[Hands] = None,
    ) -> np.ndarray:
        # Mark the image as not writeable to pass by reference, to improve performance
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        face_detections, face_names, hand_detections = [None] * 3

        if face_detector:
            face_detections = face_detector.process(image).detections

            if face_recognizer and face_detections:  # Recognition is slow
                image_rows, image_cols, _ = image.shape
                face_locations = [
                    self.convert_face_location(f.location_data, image_cols, image_rows) for f in face_detections
                ]
                face_detections = [f for i, f in enumerate(face_detections) if face_locations[i] != NOT_A_FACE]
                face_locations = [f for f in face_locations if f != NOT_A_FACE]
                face_encodings = fr.face_encodings(image, face_locations)
                face_names = [face_recognizer.recognize(face_encoding) for face_encoding in face_encodings]

        if hand_detector:
            hand_detections = hand_detector.process(image).multi_hand_landmarks

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if face_names:
            self.draw_faces_and_names(image, face_locations, face_names)
        elif face_detections:
            for face in face_detections:
                drawing_utils.draw_detection(image, face)

        if hand_detections:
            for hand in hand_detections:
                drawing_utils.draw_landmarks(
                    image=image,
                    landmark_list=hand,
                    # connections = HAND_CONNECTIONS,
                    # landmark_drawing_spec = drawing_styles.get_default_hand_landmarks_style(),
                    # connection_drawing_spec = drawing_styles.get_default_hand_connections_style()
                )
        return image

    @staticmethod
    def convert_face_location(loc: LocationData, image_cols: int, image_rows: int) -> FaceLocation:
        """Convert face location from `mediapipe.framework.formats.location_data_pb2.LocationData`
        to a tuple in css (top, right, bottom, left) order which `face_recognition` uses.
        """
        box = loc.relative_bounding_box
        # NOTE: not sure why got 'NoneType' error  even `box` is not None
        # TODO: figure out the reason and fix
        try:
            x_min, y_min = drawing_utils._normalized_to_pixel_coordinates(box.xmin, box.ymin, image_cols, image_rows)
            x_max, y_max = drawing_utils._normalized_to_pixel_coordinates(
                (box.xmin + box.width), (box.ymin + box.height), image_cols, image_rows
            )
        except TypeError as e:
            return NOT_A_FACE
        return (y_min, x_max, y_max, x_min)

    @staticmethod
    def draw_faces_and_names(image: np.ndarray, face_locations: List[FaceLocation], names: List[str]):
        """Draw faces and names on the image."""
        for face_location, name in zip(face_locations, names):
            (top, right, bottom, left) = face_location
            cv2.rectangle(image, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.rectangle(image, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            cv2.putText(image, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
