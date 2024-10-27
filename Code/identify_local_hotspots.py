import cv2
import numpy as np
import os

# Load the image
image = cv2.imread("D:\Dermatitis Thesis\Data\Input_Image_2.jpeg")

# Convert the image from BGR to HSV color space
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Define skin color range (example values, adjust as needed)
lower_skin = np.array([0, 20, 70])
upper_skin = np.array([20, 255, 255])

# Define the range of red color in HSV
# lower_red = np.array([0, 50, 50])
# upper_red = np.array([10, 255, 255])


# Create a mask for skin color
skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)

# Calculate average hue value of skin color regions
avg_hue = np.mean(hsv[:, :, 0][skin_mask > 0])

# Set a lower benchmark for redness (adjust as needed)
redness_threshold = avg_hue - 5

# Create a mask for red color
# mask1 = cv2.inRange(hsv, lower_red, upper_red)
# Create a mask for potential redness
redness_mask = (hsv[:, :, 0] < redness_threshold).astype(np.uint8) * 255


# lower_red = np.array([160, 50, 50])
# upper_red = np.array([180, 255, 255])
# mask2 = cv2.inRange(hsv, lower_red, upper_red)

# Bitwise-AND mask and original image
result = cv2.bitwise_and(image, image, mask=redness_mask)

# mask = mask1 + mask2

# Bitwise-AND mask and original image
# result = cv2.bitwise_and(image, image, mask=mask)

# cv2.imshow("Original Image", image)
cv2.imwrite(os.cwd()+"Output\Image_2_check_1.jpg", result)

