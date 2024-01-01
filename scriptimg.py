import cv2
import numpy as np


def unsharp_mask(image, sigma=1.0, strength=1.5):
    blurred = cv2.bilateralFilter(image, 2, 75, 75)
    sharpened = cv2.addWeighted(image, 1.0 + strength, blurred, -strength, 0)
    return sharpened


# Read the video file
video_path = "video.mp4"
cap = cv2.VideoCapture(video_path)


while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))
    blue = frame[:, :, 2]
    green = frame[:, :, 1]
    red = frame[:, :, 0]
    # combine images with different weights
    combined = cv2.addWeighted((blue), 0.05, green, 0.66, 0)
    combined = cv2.addWeighted(combined, 0.7, red, 0.52, 0)
    # display the combined image

    # sharpen the combined image
    kernel = np.array(
        [
            [-1, -1, -1, -1, -1],
            [-1, 1, 2, 1, -1],
            [-1, 2, 4.9, 2, -1],
            [-1, 1, 2, 1, -1],
            [-1, -1, -1, -1, -1],
        ]
    )
    
    sharpened = cv2.filter2D(combined, -1, kernel)
    sharpened = cv2.bilateralFilter(sharpened, 2, 75, 75)

    cv2.imshow("sharpened", sharpened)


    # cv2.imshow('ERODED', ERODED)
    # #display the original image
    # cv2.imshow('original', frame)
    # sharpen the dilated image
    # kernel2 = np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]])
    # sharpened_DIL = cv2.filter2D(DILATED, -1, kernel2)

    sharpened = cv2.bilateralFilter(sharpened, 6, 75, 75)
    sharpened = cv2.bitwise_not(sharpened)
    cv2.imshow("sharpened_DIL", sharpened)
    # otsu threshold the sharpened image
    ret, thresh = cv2.threshold(sharpened, 0, 255, cv2.THRESH_OTSU)
    cv2.imshow("thresh", thresh)
    # # use a hough transform to find the lines in the image
    # lines = cv2.HoughLinesP(edgy, 1, np.pi / 180, 10, 5, 10)
    # draw the lines on the original image
    # if lines is not None:
    #     for line in lines:
    #         x1, y1, x2, y2 = line[0]
    #         cv2.line(frame, (x1,y1), (x2,y2), (0,255,0), 2)
    # #display the image with the detected lines
    # cv2.imshow('frame', frame)
    edges = cv2.Canny(sharpened, 20, 150)
    cv2.imshow("edges", edges)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
