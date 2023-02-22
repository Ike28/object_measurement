import cv2


def get_contours_from_image(image, threshold=None, show=False):
    if threshold is None:
        threshold = [100, 100]
    image_grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_blur = cv2.GaussianBlur(image_grayscale, (5, 5), 1)
    canny_detection = cv2.Canny(image_blur, threshold[0], threshold[1])
    if show:
        cv2.imshow('Canny', cv2.resize(canny_detection, (0, 0), None, 0.5, 0.5))
