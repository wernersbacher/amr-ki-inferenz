from email.mime import image
import glob
import cv2
import dataAugmentater
import matplotlib.pyplot as plt

def load_images_from_directory(directory):
	images = []
	for file in glob.glob(directory):
		images.append(file)

	return images

def my_imread(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def image_preprocess(image):
    height, _, _ = image.shape
    image = image[int(height/3):,:,:]  # remove top third of the image, as it is not relavant for lane following
    image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)  # Nvidia model said it is best to use YUV color space
    image = cv2.GaussianBlur(image, (3,3), 0)
    image = cv2.resize(image, (200,66)) # input image size (200,66) Nvidia model
    image = image / 255 # normalizing, the processed image becomes black for some reason.  do we need this?
    return image

def augment_image(image, steering_angle):
    return dataAugmentater.random_augment(image, steering_angle)

def show_image(image):
    plt.imshow(image)
    plt.show()
