import time
import os
import cv2
import numpy as np
from keras import models
import matplotlib.pyplot as plt
import imageEditor
import random

FILE_FORMAT = ".png"
INPUT_DATA_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "images", "output", '*' + FILE_FORMAT)
INPUT_AUG_DATA_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "images", "outputAug", '*' + FILE_FORMAT)
model_name = "NVidiaSingle_nvidiaSingle-13-06-2022-11_25_53.h5"
model_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "models", model_name)
number_of_runs = 10
# filenames = next(walk(test_path), (None, None, []))[2]  # [] if no file

model = models.load_model(model_path)
images = imageEditor.load_images_from_directory(INPUT_DATA_DIR)


def get_richtung(angle):
    richtung = "geradeaus"

    if angle < -0.7:
        richtung = "Stark Links"
    elif angle < -0.2:
        richtung = "Links"
    elif angle < 0:
        richtung = "Leicht Links"
    elif angle > 0.7:
        richtung = "Stark Rechts"
    elif angle > 0.2:
        richtung = "Rechts"
    elif angle > 0:
        richtung = "Leicht Rechts"

    return richtung


for i in range(number_of_runs):
    random_index = random.randint(0, len(images) - 1)
    image_path = images[random_index]
    image = cv2.imread(image_path)
    _, idi, throttle, steering_angle, times = image_path.split("_")
    steering_angle = float(steering_angle)

    preprocessed_img = imageEditor.image_preprocess(image)
    img_array = np.asarray(preprocessed_img)
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict([img_array])
    plt.imshow(image)
    diff = abs(prediction[0][0] - steering_angle)
    richtung = get_richtung(prediction[0][0])

    prediction_string = "Prediction: {:.4f}, Steering Angle: {:.4f}, Differenz: {:.4f}, {}".format(prediction[0][0],
                                                                                                   steering_angle, diff,
                                                                                                   richtung)
    plt.text(0, -20, prediction_string)
    plt.show()
