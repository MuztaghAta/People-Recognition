from pathlib import Path
from typing import List

import face_recognition as fr
import numpy as np


class FaceRecognizer:
    """Face recognition.

    Use either `encode()` or `encode_face_images_in_dir()` to build face encodings which will be used for recognizing
    faces with  `recognize()`.

    TODO:
    - try openface directly
    - try other classification methods for recognition
    - make recognition faster and more accurate
    - evaluate recognition

    Ref:
    - face_recognition
        - (doc)[https://face-recognition.readthedocs.io/en/latest/index.html]
        - (code)[https://github.com/ageitgey/face_recognition]
        - (article)[https://medium.com/@ageitgey/machine-learning-is-fun-part-4-modern-face-recognition-with-deep-learning-c3cffc121d78]

    - openface
        - (doc)[http://cmusatyalab.github.io/openface/#overview]
        - (code)[https://github.com/cmusatyalab/openface]
    """

    def __init__(self) -> None:
        pass

    def encode(self, images: List[np.ndarray], names: List[str]) -> "FaceRecognizer":
        """Create encoding for the face in each image. Make sure:
        1. each image contains only one face.
        2. each name is correctly mapped to each face.
        """
        assert len(images) == len(names)
        print("Encoding faces ...")
        self.face_encodings_ = [fr.face_encodings(img)[0] for img in images]
        self.names_ = names
        return self

    def encode_face_images_in_dir(self, dir: Path, ext: str = "*") -> "FaceRecognizer":
        """Load face images fro 'dir'. Assume the file name of each image is the name of the person in the image."""
        self.face_encodings_, self.names_ = [], []
        print("Encoding faces ...")
        for path in dir.glob(f"*.{ext}"):
            image = fr.load_image_file(path)
            encoding = fr.face_encodings(image)[0]
            self.face_encodings_.append(encoding)
            self.names_.append(path.stem)
        return self

    def recognize(self, face_encoding: np.ndarray) -> str:
        """Recognize a face by comparing 'face_encoding' with labeled face encodings."""
        name = "Unknown"
        matches = fr.compare_faces(self.face_encodings_, face_encoding)
        face_distances = fr.face_distance(self.face_encodings_, face_encoding)
        if len(matches) > 0 and len(face_distances) > 0:
            best_match_index = np.argmin(face_distances)
            name = self.names_[best_match_index] if matches[best_match_index] else name
        return name

    def evaluate(self):
        """Evaluate the fit FaceRecognizer.
        TODO: implement.
        """
        pass
