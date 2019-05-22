import sys
import cv2
from keras.models import load_model
import numpy as np
from utils import preprocess_input

import tensorflow as tf
import argparse

# parameters for loading images
parser = argparse.ArgumentParser(description="Script to do inference on Blink Detection model")
parser.add_argument("input_image_path",
                    help="Path to input test image.")
parser.add_argument("face_detection_model_path",
                    help="Path to face detection model.")
parser.add_argument("blink_detect_model_path",
                    help="Path to blink detection model.")
FLAGS, unparsed = parser.parse_known_args()
if unparsed:
    print("Usage: %s <input test image>")
    exit()

image_path = FLAGS.input_image_path

# loading models
#face_detection_model_path = '../haarcascade_frontalface_default.xml'
#blink_detect_model_path = '../trained_models/cew_blink_detect.70-1.00.hdf5'
face_detection_model = cv2.CascadeClassifier(FLAGS.face_detection_model_path)
blink_detect_classifier = load_model(FLAGS.blink_detect_model_path, compile=False)
blink_detect_target_size = blink_detect_classifier.input_shape[1:3]
blink_detect_classifier.summary()
# loading images
rgb_image = cv2.imread(image_path)
faces = face_detection_model.detectMultiScale(rgb_image, 1.3, 5)

#Output Lables
test_labels = {0: 'Not Blinked', 1: 'Blinked'}

#print("faces: ",faces)
if len(faces) is 0:
    print("Could not detect any face in the image provided, please try another one...")
    exit()
for face_coordinates in faces:
    x, y, w, h = face_coordinates
    rgb_face = rgb_image[y:y+h, x:x+w]

    try:
        rgb_face = cv2.resize(rgb_face, (blink_detect_target_size))
    except:
        continue
    rgb_face = preprocess_input(rgb_face, False)
    rgb_face = np.expand_dims(rgb_face, 0)

    blink_prediction = blink_detect_classifier.predict(rgb_face)
    blink_label_arg = np.argmax(blink_prediction)
    blink_output = test_labels[blink_label_arg]
    print(blink_output)
    cv2.rectangle(rgb_image, (x, y), (x + w, y + h), (0, 255, 0), 3)
    cv2.putText(rgb_image, blink_output, (x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                3, (0, 255, 0), 3, cv2.LINE_AA)
#cv2.namedWindow('Test Image', cv2.WINDOW_NORMAL)
cv2.imshow('Test Image',rgb_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
