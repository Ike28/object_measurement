import cv2
import utils

webcam = False
path = 'sample.jpg'

# set up webcam capture
cap = cv2.VideoCapture(0)
cap.set(10, 160)
cap.set(3, 1920)
cap.set(4, 1080)

# set up background dimensions
scale = 2
width_a4 = 210 * scale
height_a4 = 297 * scale

while True:
    if webcam:
        success, img = cap.read()
    else:
        img = cv2.imread(path)

    img, contours = utils.get_contours_from_image(img, minArea=50000, filt=4)

    if len(contours) > 0:
        max_contour = contours[0][2]
        image_un_warped = utils.un_warp_image(img, max_contour, width_a4, height_a4)
        contours_image, object_contours = utils\
            .get_contours_from_image(image_un_warped, minArea=2000, filt=4, threshold=[50, 50])
        cv2.imshow('A4', contours_image)

        if len(contours) > 0:
            for obj in object_contours:
                cv2.polylines(contours_image, [obj[2]], True, (0, 255, 0), 2)

    img = cv2.resize(img, (0, 0), None, 0.5, 0.5)
    cv2.imshow('Original', img)
    cv2.waitKey(1)

