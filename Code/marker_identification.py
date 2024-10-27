# import cv2
# import numpy as np

# def keep_marked_portions(image_path):
#     # Read the image
#     image = cv2.imread(image_path)

#     # Convert to grayscale
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#     # Threshold the grayscale image to create a binary mask
#     _, binary_mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

#     # Find contours (shapes) in the binary mask
#     contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#     # Create an empty mask to store the marked areas
#     marked_area_mask = np.zeros_like(binary_mask)

#     # Draw filled contours (pen outlines) on the mask
#     for contour in contours:
#         cv2.drawContours(marked_area_mask, [contour], -1, 255, -1)

#     # Invert the mask (to keep the marked areas)
#     marked_area_mask = cv2.bitwise_not(marked_area_mask)

#     # Apply the mask to the original image
#     result_image = cv2.bitwise_and(image, image, mask=marked_area_mask)

#     # Save the result
#     cv2.imwrite("D:\Dermatitis Thesis\Data\Output\out_image_1.jpeg", result_image)


import cv2
import numpy as np


def identify_red_spots(image_path):
    # Read the image
    image = cv2.imread(image_path)

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Calculate the gradient magnitude (Sobel operator)
    # gradient_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    # gradient_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    # gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)

    # Normalize the gradient magnitude to 0-255 range
    # gradient_magnitude = cv2.normalize(
    #     gradient_magnitude, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U
    # )

    # Threshold the gradient magnitude to create a binary mask
    _, binary_mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(
        binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    pen_outline_mask = np.zeros_like(binary_mask)

    for contour in contours:
        cv2.drawContours(pen_outline_mask, [contour], -1, 255, -1)
    # binary_mask = cv2.bitwise_not(binary_mask)
    # pen_outline_mask = np.zeros_like(binary_mask)

    # Dilate the mask to include a bit of the surrounding areas
    # kernel = np.ones((15, 15), np.uint8)
    # binary_mask = cv2.dilate(binary_mask, kernel, iterations=1)

    # Apply the mask to the original image
    result_image = cv2.bitwise_and(image, image, mask=pen_outline_mask)

    # Save the result
    cv2.imwrite(
        "D:\Dermatitis Thesis\Data\Output\pen_markings_and_surrounding.jpg",
        gray,
    )


# Example usage
image_path = "D:\Dermatitis Thesis\Data\Image_1.jpg"
identify_red_spots(image_path)

# Print a success message
print("Red spots and surrounding areas saved as red_spots_and_surrounding.jpg")

# # Example usage
# image_path = "D:\Dermatitis Thesis\Data\Image_1.jpg"
# keep_marked_portions(image_path)
