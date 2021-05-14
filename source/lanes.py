import cv2
import numpy as np

test_image = cv2.imread('../image/test_image.jpg')

# Copy our image array into a new variable. Thus, the changes we make will not affect the actual image.
image = np.copy(test_image)
grayscale_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
blurred_image = cv2.GaussianBlur(grayscale_image, (5, 5), 0)
canny_image = cv2.Canny(blurred_image, 50, 150)

cv2.imshow('Image', canny_image)
cv2.waitKey(0)
