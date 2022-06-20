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
#filenames = next(walk(test_path), (None, None, []))[2]  # [] if no file

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
    steering_angle = float(image_path.split('_')[-1][:-4])
    preprocessed_img = imageEditor.image_preprocess(image)
    img_array = np.asarray(preprocessed_img)
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict([img_array])
    plt.imshow(image)
    diff = abs(prediction[0][0] - steering_angle)
    richtung = get_richtung(prediction[0][0])

    prediction_string = "Prediction: {:.4f}, Steering Angle: {:.4f}, Differenz: {:.4f}, {}".format(prediction[0][0], steering_angle, diff, richtung)
    plt.text(0, -20, prediction_string)
    plt.show()


'''for file in filenames:
    _, idi, throttle, steering, times = file.split("_")

    im = cv2.imread(os.path.join(test_path, file))
    im = cv2.cvtColor(im, cv2.COLOR_BGR2YUV)
    print(im.shape)
    #im = im[50:-50, :]
    im = cv2.resize(im, (200, 66), interpolation=cv2.INTER_AREA)

    # rotate
    im = cv2.transpose(im)
    im = cv2.flip(im, flipCode=1)

    img_array = np.array(im)
    img_array = np.expand_dims(img_array, axis=0)
    print(img_array.shape)
    time.sleep(2)

    prediction = model.predict(img_array)
    print(f"{prediction}, {steering}")

    if prediction < -0.1:
        print("Left")
    elif prediction > 0.1:
        print("Right")
    else:
        print("No steering")
    """
    plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    plt.show()"""


    #time.sleep(2) '''
