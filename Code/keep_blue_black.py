import cv2
import numpy as np

def keep_blue_or_black_lines(image_path, color="blue"):
    # Read the image
    image = cv2.imread(image_path)

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Create a binary mask for blue or black lines
    if color == "blue":
        # Filter by blue channel
        blue_mask = image[:, :, 0] > 100  # Adjust threshold as needed
        binary_mask = blue_mask.astype(np.uint8) * 255
    elif color == "black":
        # Filter by grayscale intensity
        _, binary_mask = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
    else:
        print("Invalid color specified. Choose 'blue' or 'black'.")
        return

    # Apply the mask to the original image
    result_image = cv2.bitwise_and(image, image, mask=binary_mask)

    # Save the result
    output_filename = f"D:\Dermatitis Thesis\Data\Output\{color}_lines_only.jpg"
    cv2.imwrite(output_filename, result_image)

    print(f"{color.capitalize()} lines saved as {output_filename}")

# Example usage
image_path = "D:\Dermatitis Thesis\Data\Image_1.jpg"
keep_blue_or_black_lines(image_path, color="black")  # Specify 'blue' or 'black'
