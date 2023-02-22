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
             list containing elements of structure:
                [vertices, area, approximated contour shape, bounding rectangle, contour as vector of points]
    """
    if threshold is None:
        threshold = [100, 100]
    image_grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_blur = cv2.GaussianBlur(image_grayscale, (5, 5), 1)
    canny_detection = cv2.Canny(image_blur, threshold[0], threshold[1])

    kernel = np.ones((5, 5))
    image_dilated = cv2.dilate(canny_detection, kernel, iterations=1)
    image_threshold = cv2.erode(image_dilated, kernel, iterations=1)

    if show:
        #image_show = cv2.resize(image_threshold, (0, 0), None, 0.5, 0.5)
        cv2.imshow('Canny', image_threshold)

    contours, hierarchy = cv2.findContours(image_threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contours_res = []
    for c in contours:
        area = cv2.contourArea(c)
        if area > minArea:
            perimeter = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.09*perimeter, True)
            bounding = cv2.boundingRect(approx)
            if filt > 0:
                if len(approx) == filt:
                    contours_res.append([len(approx), area, approx, bounding, c])
            else:
                contours_res.append([len(approx), area, approx, bounding, c])

    contours_res = sorted(contours_res, key=lambda x: x[1], reverse=True)
    if draw:
        for c in contours_res:
            cv2.drawContours(image, c[4], -1, (0, 0, 255), 3)

    return image, contours_res


def re_order_points(points):
    """
    Reorders points of a 4-vertex rectangle to form convex rectangle
    :param points: input points
    :return: reordered points
    """
    points_new = np.zeros_like(points)
    points = points.reshape((4, 2))
    add = points.sum(1)
    points_new[0] = points[np.argmin(add)]
    points_new[3] = points[np.argmax(add)]

    diff = np.diff(points, axis=1)
    points_new[1] = points[np.argmin(diff)]
    points_new[2] = points[np.argmax(diff)]

    return points_new


def un_warp_image(image, points, width, height, padding=10):
    """
    Un-warps an object from an image to top-down perspective
    :param image: input image
    :param points: object points
    :param width: width of the object
    :param height: height of the object
    :param padding: size of object edge padding in pixels (to reduce irregularities)
    :return:
    """
    points = re_order_points(points)
    points_warped = np.float32(points)
    points_new = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
    matrix = cv2.getPerspectiveTransform(points_warped, points_new)
    image_un_warped = cv2.warpPerspective(image, matrix, (width, height))
    image_un_warped = image_un_warped[padding:image_un_warped.shape[0] - padding, padding:image_un_warped.shape[1] - padding]

    return image_un_warped


def find_distance(points1, points2):
    return ((points2[0] - points1[0])**2 + (points2[1] - points1[1])**2)**0.5
