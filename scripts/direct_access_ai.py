import cv2
import rospy
from geometry_msgs.msg import Twist
import tflite_runtime.interpreter as tflite
from types import SimpleNamespace
import numpy as np
import os
import time

SPEED = 0.250

RECORD = True

MODEL_NAME = "model.tflite"
MODEL_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../models", MODEL_NAME)

a = {
    "colors": True,
    "video_stream_provider": 0,
    "fps": 15,
    "width": 300,
    "height": 200,
    "camera_name": "camera",
    "frame_id": "camera",
    "buffer_queue_size": 10

}
args = SimpleNamespace(**a)


def image_preprocess(image):
    height, _, _ = image.shape
    image = image[int(height/3):, :, :]  # remove top third of the image, as it is not relavant for lane following
    #image = image[25:, :, :]  # remove top third of the image, as it is not relavant for lane following
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

        self.throttle = SPEED
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


ki = KI()



def loop_img():

    rospy.init_node('amr_ki_interferenz')
    publisher_twist = rospy.Publisher("/cmd_vel", Twist, queue_size=1)
    id = 1

    dir = "/var/recorded_data/ki_interferenz"+str(int(time.time()))
    os.mkdir(dir)

    # convert to int?
    try:
        video_channel = int(args.video_stream_provider)
    except ValueError:
        video_channel = args.video_stream_provider

    rospy.loginfo ("Loaded arguments:")
    rospy.loginfo(args)

    # Open video.
    video = cv2.VideoCapture(video_channel)
    rospy.loginfo("Publishing %s." % video_channel)

    fps_cam = video.get(cv2.CAP_PROP_FPS)

    # Get frame rate.
    try:
        fps_arg = int(args.fps)
        if 0 < fps_arg < fps_cam:
            fps = fps_arg
        else:
            fps = fps_cam
    except ValueError:
        fps = fps_cam

    rospy.loginfo(f"Watching at {fps} FPS")
    rate = rospy.Rate(fps)

    # Loop through video frames.
    while not rospy.is_shutdown() and video.grab():
        t1 = time.time()
        tmp, img_color = video.retrieve()

        # converting to gray-scale
        if not args.colors:
            img = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
        else:
            img = img_color

        if not tmp:
            rospy.logerror("Could not grab frame.")
            break

        img_out = np.empty((args.height, args.width))

        # Compute input/output aspect ratios.
        aspect_ratio_in = np.float(img.shape[1]) / np.float(img.shape[0])
        aspect_ratio_out = np.float(args.width) / np.float(args.height)

        if aspect_ratio_in > aspect_ratio_out:
            # Output is narrower than input -> crop left/right.
            rsz_factor = np.float(args.height) / np.float(img.shape[0])
            img_rsz = cv2.resize(img, (0, 0), fx=rsz_factor, fy=rsz_factor,
                                 interpolation=cv2.INTER_AREA)

            diff = int((img_rsz.shape[1] - args.width) / 2)

            if args.colors:
                img_out = img_rsz[:, diff:-diff-1, :]
            else:
                img_out = img_rsz[:, diff:-diff-1]

        elif aspect_ratio_in < aspect_ratio_out:
            # Output is wider than input -> crop top/bottom.
            rsz_factor = np.float(args.width) / np.float(img.shape[1])
            img_rsz = cv2.resize(img, (0, 0), fx=rsz_factor, fy=rsz_factor,
                                 interpolation=cv2.INTER_AREA)

            diff = int((img_rsz.shape[0] - args.height) / 2)

            if args.colors:
                img_out = img_rsz[diff:-diff-1, :, :]
            else:
                img_out = img_rsz[diff:-diff-1, :]
        else:
            # Resize image.
            img_out = cv2.resize(img, (args.width, args.height))

        steering_angle = ki.predict(img_out)
        twist_msg = convert_to_twist(SPEED, steering_angle)
        publisher_twist.publish(twist_msg)
        t2 = time.time()
        
        if RECORD:
            file_name = f"img_{id}_{SPEED}_{steering_angle:.4f}_{time.time()}.png"
            path = f"{dir}/{file_name}"
            cv2.imwrite(path, img_out)
            print(f"image written to {path}")
            id+=1
        
        tdiff = int((t2-t1)*1000)

        print(f"time loop: {tdiff}")

        rate.sleep()


if __name__ == '__main__':
    try:
        print("Starting KI")

        loop_img()

        #rospy.spin()
    except rospy.ROSInterruptException as e:
        rospy.loginfo("Goodbye.")
