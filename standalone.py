import time
from os import walk
import cv2
import numpy as np
from keras import models
import os
import matplotlib.pyplot as plt


test_path = "test2"
model_file = "YUV_nvidiaSingle-01-06-2022-03 04 54.h5"

filenames = next(walk(test_path), (None, None, []))[2]  # [] if no file

model = models.load_model(model_file)

for file in filenames:
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


    #time.sleep(2)
