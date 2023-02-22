import cv2
import numpy as np


def get_contours_from_image(image, threshold=None, show=False, minArea=1000, filt=0, draw=False):
    """
    Returns data about contours in an image using Canny detection
    :param image: input image
    :param threshold: threshold for detection
    :param show: display an image of the contours, boolean
    :param minArea: minimum area of objects
    :param filt: number of vertices for contours, int
    :param draw: draw contours onto original image
    :return: input image (after optional modifications),
             array containing
                (vertices, area, approximated contour shape, bounding rectangle, contour as vector of points)
    """
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

    contours, hierarchy = cv2.findContours(image_threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contours_res = []
    for c in contours:
        area = cv2.contourArea(c)
        if area > minArea:
            perimeter = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02*perimeter, True)
            bounding = cv2.boundingRect(approx)
            if filt > 0:
                if len(approx) == filt:
                    contours_res.append(len(approx), area, approx, bounding, c)
            else:
                contours_res.append((len(approx), area, approx, bounding, c))

    contours_res = sorted(contours_res, key=lambda x: x[1], reverse=True)
    if draw:
        for c in contours_res:
            cv2.drawContours(image, c[4], -1, (0, 0, 255), 10)

    return image, contours_res
