import cv2
import numpy as np

# Function to calculate wound metrics
def calculate_metrics(pred_mask):
    # Ensure mask is binary (0 or 255)
    pred_mask = (pred_mask > 0).astype(np.uint8) * 255

    # Find contours of the mask
    contours, _ = cv2.findContours(pred_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Get the largest contour (assuming it's the wound region)
        contour = max(contours, key=cv2.contourArea)

        # Calculate metrics
        area = cv2.contourArea(contour)  # Area in pixels
        perimeter = cv2.arcLength(contour, True)  # Perimeter in pixels
        x, y, w, h = cv2.boundingRect(contour)  # Bounding box (x, y, width, height)
        length = h  # Height of the bounding box as length
        width = w  # Width of the bounding box as width

        # Calculate circularity (to prevent division by zero)
        if perimeter == 0:
            circularity = 0
        else:
            circularity = 4 * np.pi * (area / (perimeter ** 2))

        # Return the calculated metrics
        return length, width, area, perimeter, circularity
    else:
        # If no contours are found, return 0 for all metrics
        return 0, 0, 0, 0, 0
