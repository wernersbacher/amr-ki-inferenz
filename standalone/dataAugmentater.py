import cv2
from imgaug import augmenters
import numpy as np
import random

def _zoom_augmentation(image):
	return augmenters.Affine(scale=(1, 1.3)).augment_image(image)	# zoom to 130%

def _flip_augmentation(image, steering_angle):
	flipped_image = cv2.flip(image, 1)
	steering_angle_float = float(steering_angle)
	if float(steering_angle) < 0:
		adjusted_steering_angle = abs(steering_angle_float)
	else:
		adjusted_steering_angle = 0 - abs(steering_angle_float)

	return flipped_image, adjusted_steering_angle

def _pan_augmentation(image):
	pan = augmenters.Affine(translate_percent={"x":(-0.1, 0.1), "y":(-0.1, 0.1)})
	panned_image = pan.augment_image(image)
	return panned_image

def _brightness_augmentation(image):
	brightness = augmenters.Multiply((0.7, 1.3))	# increase or decrease brightness by 30%
	edited_image = brightness.augment_image(image)
	return edited_image

def _blur_augmentation(image):
	kernel_size = random.randint(1, 5)
	blurred_image = cv2.blur(image, (kernel_size, kernel_size))
	return blurred_image

def random_augment(image, steering_angle):
    if np.random.rand() < 0.5:
        image = _pan_augmentation(image)
    if np.random.rand() < 0.5:
        image = _zoom_augmentation(image)
    if np.random.rand() < 0.5:
        image = _blur_augmentation(image)
    if np.random.rand() < 0.5:
        image = _brightness_augmentation(image)
    image, steering_angle = _flip_augmentation(image, steering_angle)
    
    return image, steering_angle