from types import FrameType
import cv2
import numpy as np
import matplotlib.pyplot as plt

def make_coordinates(image, line_parameters):
    slope, intercept = line_parameters
    y1 = image.shape[0]     # Bottom of the image
    y2 = int(y1 * (3 / 5))  # Slightly lower than the middle
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return [[x1, y1, x2, y2]]

def average_slope_intercept(image, lines):
    left_fit = []
    right_fit = []
    if lines is None:
        return None
    for line in lines:
        for x1, y1, x2, y2 in line:
            fit = np.polyfit((x1, x2), (y1, y2), 1)
            slope = fit[0]
            intercept = fit[1]
            if slope < 0: # y is reversed in image
                left_fit.append((slope, intercept))
            else:
                right_fit.append((slope, intercept))
    left_fit_average = np.average(left_fit, axis=0)
    right_fit_average = np.average(right_fit, axis=0)
    left_line = make_coordinates(image, left_fit_average)
    right_line = make_coordinates(image, right_fit_average)
    return [left_line, right_line]

def canny(image):
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    kernel = 5
    blurred_image = cv2.GaussianBlur(grayscale_image, (kernel, kernel), 0)
    canny_image = cv2.Canny(blurred_image, 50, 150)
    return canny_image

def region_of_interest(image):
    height = image.shape[0]
    mask = np.zeros_like(image)
    polygons = np.array([[(200, height), (1100, height), (550, 250)]], np.int32)    
    cv2.fillPoly(mask, polygons, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

def display_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
    return line_image


# test_image = cv2.imread('../image/test_image.jpg')

# # Copy our image array into a new variable. Thus, the changes we make will not affect the actual image.
# image = np.copy(test_image)

# canny_image = canny(image)
# cropped_image = region_of_interest(canny_image)
# lines = cv2.HoughLinesP(cropped_image, 2, np.pi / 180, 100, np.array([]), minLineLength=40, maxLineGap=5)
# averaged_lines = average_slope_intercept(image, lines)
# line_image = display_lines(image, averaged_lines)
# image_with_lines = cv2.addWeighted(image, 0.8, line_image, 1, 1)

# cv2.imshow('Result', image_with_lines)
# cv2.waitKey(0)


cap = cv2.VideoCapture('../video/test_video.mp4')
while (cap.isOpened()):
    # Returns two values which we can unpack. First value is a boolean and second is current frame of the video.
    _, frame = cap.read()
    canny_image = canny(frame)
    cropped_image = region_of_interest(canny_image)
    lines = cv2.HoughLinesP(cropped_image, 2, np.pi / 180, 100, np.array([]), minLineLength=40, maxLineGap=5)
    averaged_lines = average_slope_intercept(frame, lines)
    line_image = display_lines(frame, averaged_lines)
    image_with_lines = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
    cv2.imshow('Video', image_with_lines)
    # waitKey returns a 32bit integer value.
    # Common trick is to apply bitwise AND operation with the hexadecimal constant to mask our integer to eight bits
    # We are comparing this to the numeric encoding of the 'q' character to break the loop:
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
