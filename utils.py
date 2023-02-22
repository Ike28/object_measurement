import cv2
import numpy as np


def get_contours_from_image(image, threshold=None, show=False):
    if threshold is None:
        threshold = [100, 100]
    image_grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_blur = cv2.GaussianBlur(image_grayscale, (5, 5), 1)
    canny_detection = cv2.Canny(image_blur, threshold[0], threshold[1])

    kernel = np.ones((5, 5))
    image_dilated = cv2.dilate(canny_detection, kernel, iterations=3)
    image_threshold = cv2.erode(image_dilated, kernel, iterations=2)

    if show:
        image_show = cv2.resize(image_threshold, (0, 0), None, 0.5, 0.5)
        cv2.imshow('Canny', image_show)
