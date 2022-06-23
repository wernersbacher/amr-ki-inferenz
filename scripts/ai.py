#!/usr/bin/env python3

import rospy
import cv2
import time
import os
import numpy as np
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool
from sensor_msgs.msg import Image
import tflite_runtime.interpreter as tflite

from cv_bridge import CvBridge, CvBridgeError

bridge = CvBridge()

FPS_INTERFERENZ = 0  # set 0 to save every image!

MODEL_NAME = "model.tflite"
MODEL_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../models", MODEL_NAME)


def image_preprocess(image):
    height, _, _ = image.shape
    image = image[int(height / 3):, :, :]  # remove top third of the image, as it is not relavant for lane following
    image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)  # Nvidia model said it is best to use YUV color space
    image = cv2.GaussianBlur(image, (3, 3), 0)
    image = cv2.resize(image, (200, 66))  # input image size (200,66) Nvidia model
    image = image / 255  # normalizing, the processed image becomes black for some reason.  do we need this?
    return image


def convert_to_twist(throttle, steering):

    twist_msg = Twist()
    twist_msg.linear.x = float(throttle)
    twist_msg.linear.y = 0
    twist_msg.linear.z = 0
    twist_msg.angular.x = 0
    twist_msg.angular.y = 0
    twist_msg.angular.z = float(steering)

    return twist_msg


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


class KI:

    def __init__(self):
        self.interpreter = tflite.Interpreter(model_path=MODEL_PATH)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        self.throttle = 0.27
        self.steering = 0

        self.last_image_saved = 0

    def predict(self, img):
        preprocessed_img = image_preprocess(img)
        img_array = np.asarray(preprocessed_img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = np.float32(img_array)

        self.interpreter.set_tensor(self.input_details[0]['index'], img_array)
        self.interpreter.invoke()

        prediction = self.interpreter.get_tensor(self.output_details[0]['index'])

        steering_angle = prediction[0][0]

        return steering_angle


class KIConn:

    def __init__(self) -> None:

        number_of_runs = 10
        # filenames = next(walk(test_path), (None, None, []))[2]  # [] if no file

        print("Loading model")

        self.KI = KI()

        print("Done loading model")

        rospy.loginfo("Setting Up the Node...")
        rospy.init_node('amr_ki_interferenz')

        # --- Create the Subscriber to Twist commands
        self.publisher_twist = rospy.Publisher("/cmd_vel", Twist, queue_size=1)
        self.subscriber_img = rospy.Subscriber("/camera/image_raw", Image, self.update_img)

        self.throttle = 0.27
        self.steering = 0

        self.last_image_saved = 0

    def update_twist(self):

        twist_msg = convert_to_twist(self.throttle, self.steering)
        self.publisher_twist.publish(twist_msg)

    def update_img(self, msg):

        now = time.time()

        if FPS_INTERFERENZ != 0 and now - self.last_image_saved < 1/FPS_INTERFERENZ:
            rospy.logdebug("Skipping frame.")
            return

        self.last_image_saved = now

        rospy.logdebug(f"Received an image, processing...")

        image = bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

        steering_angle = self.KI.predict(image)
        t2 = time.time()
        rospy.loginfo(f"Calculated steering angle of {steering_angle} ({get_richtung(steering_angle)}) "
                      f"(took {int((t2-now)*1000)}ms) ")
        self.steering = steering_angle

        self.update_twist()


if __name__ == '__main__':
    try:
        print("Starting KI")

        KIConn()
        rospy.spin()
    except rospy.ROSInterruptException as e:
        rospy.loginfo("Goodbye.")
