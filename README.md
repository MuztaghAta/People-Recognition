# Description

The intention of this project is to explore the possibility of recognizing people in meeting rooms in online meetings. Being able to recognize people in meeting rooms could unlock a number of real-time services (e.g. show the people in the meeting room individually on the Teams meeting interface and signal hand-up for the correct person on the interface whenever hand-up is detected in the meeting room). In addition, it could enable a number of post analytics with knowing who joined the meeting from the room.

Right now, people recognition is done through face recognition. Voice recognition could work too, but will probably be less timely if the people do not speak up early. Nevertheless, it could enhance the recognition results. Hand detection is enabled too without knowing to whom the detected hands belong. It would also be interesting to detect other emotions, gestures and movements to unlock more services.

## Requirements

The main frameworks for this project are opencv, face_recognition, and mediapipe. Create a new virtual environment and activate it, then checkout to the project root `people_recognition`, and install the dependencies:
```
python -m pip install -r requirements.txt
```

## Run people recognition in camera video

To be able to recognize any person in the video, a labeled face image of that person is required. The `PeopleRecognizer` will encode the labeled face images you collect and use them to recognize people by comparing the encodings of the labeled face images and encodings of detected people from the video. Once you got the "training data", change the image dir in script `sources/dev/people_recognition/run.py` and then run the following script to see the people recognition with you camera video (you may need to change the index of the camera in the script).
