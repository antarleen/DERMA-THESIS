import cv2
import pytesseract
import numpy as np

def preprocess_image(image_path):
    # Read the image
    image = cv2.imread("D:\Dermatitis Thesis\Data\Image_5.jpeg")
    image = cv2.resize(image,  (0, 0),fx=0.5, fy=0.5)
    # Convert the image to grayscale
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply GaussianBlur to reduce noise and improve contour detection
    # blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Use the HoughCircles method to detect circles
    # circles = cv2.HoughCircles(
    #     blurred,
    #     cv2.HOUGH_GRADIENT,
    #     dp=1,
    #     minDist=50,
    #     param1=50,
    #     param2=30,
    #     minRadius=10,
    #     maxRadius=100
    # )
    
    # if circles is not None:
    #     circles = np.uint16(np.around(circles))
        
    #     # Draw circles on the original image
    #     for circle in circles[0, :]:
    #         cv2.circle(image, (circle[0], circle[1]), circle[2], (0, 255, 0), 2)
            
    #     # Extract numbers using pytesseract
    #     for circle in circles[0, :]:
    #         x, y, r = circle[0], circle[1], circle[2]
    #         roi = gray[y - r:y + r, x - r:x + r]
    #         _, binary_roi = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    #         number = pytesseract.image_to_string(binary_roi, config='--psm 6')
            
    #         # Display the extracted number
    #         print("Number in circle:", number)
            
    # Display the processed image
    # cv2.imshow("Processed Image", blurred)
    
    # Convert the image from BGR to RGB
    # rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Convert the image from BGR to YCbCr
    ycbcr_image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)

    # Define a lower and upper range for skin tones in YCbCr color space
    lower_skin = np.array([0, 135,102], dtype=np.uint8)
    upper_skin = np.array([255, 180, 255], dtype=np.uint8)


    # Define a lower and upper range for the red color
    # lower_red = np.array([130, 0, 11], dtype=np.uint8)
    # upper_red = np.array([100, 100, 255], dtype=np.uint8)
    # Define a lower and upper range for skin tones
    # lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    # upper_skin = np.array([20, 255, 255], dtype=np.uint8)


    # Create a mask using the inRange function to extract only the red color
    red_mask = cv2.inRange(ycbcr_image, lower_skin, upper_skin)

    # Apply the mask to the original image
    result_image = cv2.bitwise_and(image, image, mask=red_mask)

    # Convert the result back to BGR for display or further processing
    result_image_bgr = cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(result_image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    # Display the original and result images
    cv2.imshow('Original Image', image)
    cv2.imshow('Only Red Color', blurred)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Replace 'your_image_path.jpg' with the path to your image
preprocess_image('your_image_path.jpg')