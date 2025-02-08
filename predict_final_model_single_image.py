import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from calculate_metrics import calculate_metrics  # Import the metrics calculation function
from PIL import Image, ImageDraw, ImageFont  # Import Pillow for advanced text rendering

# Paths
MODEL_PATH = 'outputs/unet/unet_model.h5'
INPUT_IMAGE_PATH = 'outputs/input/1.png'  # Replace with your input image path
PREDICTED_MASK_PATH = 'outputs/single/predicted_mask.png'
COLORED_COMPARISON_PATH = 'outputs/single/colour_comparison.png'
COMPARISON_IMAGE_PATH = 'outputs/single/comparison_image.png'

# Load the trained model
model = load_model(MODEL_PATH, compile=False)

# Image size used during model training
IMAGE_SIZE = (256, 256)

# Function to process a single image
def process_single_image():
    # Load and resize the input image
    original_image = load_img(INPUT_IMAGE_PATH)
    original_image_resized = load_img(INPUT_IMAGE_PATH, target_size=IMAGE_SIZE)
    original_image_array = img_to_array(original_image_resized) / 255.0
    
    # Predict mask
    pred_mask = model.predict(np.expand_dims(original_image_array, axis=0))[0]
    pred_mask = (pred_mask > 0.5).astype(np.uint8)  # Binarize mask
    
    # Save predicted mask
    pred_mask_resized = cv2.resize(pred_mask, (original_image.size[0], original_image.size[1]))  # Resize to original size
    cv2.imwrite(PREDICTED_MASK_PATH, pred_mask_resized * 255)
    
    # Convert the original image from RGB to BGR (since OpenCV uses BGR format)
    original_image_bgr = cv2.cvtColor(np.array(original_image), cv2.COLOR_RGB2BGR)
    
    # Create predicted mask in 3 channels (grayscale to RGB-like BGR for OpenCV)
    pred_mask_bgr = np.repeat(pred_mask_resized[:, :, np.newaxis], 3, axis=2) * 255
    
    # ------------------- Color-coded overlay -------------------
    # Create a translucent red and blue overlay
    red_overlay = np.zeros_like(original_image_bgr, dtype=np.uint8)
    blue_overlay = np.zeros_like(original_image_bgr, dtype=np.uint8)
    
    # Set red for mask areas (where the predicted mask is white)
    red_overlay[:, :, 2] = 255  # Red channel to max (BGR format)
    
    # Set blue for non-mask areas (where the predicted mask is black)
    blue_overlay[:, :, 0] = 255  # Blue channel to max (BGR format)
    
    # Create the translucent overlays based on the predicted mask
    alpha_red = 0.5  # Transparency factor for red overlay
    alpha_blue = 0.5  # Transparency factor for blue overlay
    color_overlay = np.where(pred_mask_resized[:, :, np.newaxis], 
                             cv2.addWeighted(original_image_bgr, 1, red_overlay, alpha_red, 0),
                             cv2.addWeighted(original_image_bgr, 1, blue_overlay, alpha_blue, 0))
    
    # Save the color-coded comparison image
    cv2.imwrite(COLORED_COMPARISON_PATH, color_overlay)

    # Calculate metrics
    length, width, area, perimeter, circularity = calculate_metrics(pred_mask_resized)

    # Create comparison image (original + predicted mask + color overlay side by side)
    comparison_image = np.hstack((original_image_bgr, pred_mask_bgr, color_overlay))

    # Convert comparison image from OpenCV to Pillow for text rendering
    comparison_image_pil = Image.fromarray(cv2.cvtColor(comparison_image, cv2.COLOR_BGR2RGB))

    # Create a blank space for writing metrics below the image with padding for spacing
    padding_height = 100  # Space above the metrics text
    blank_space_height = 110  # Fixed height for the black space below the images
    blank_space = Image.new("RGB", (comparison_image.shape[1], blank_space_height), (0, 0, 0))
    
    # Load a font (use a TrueType font, you can adjust the path and size as needed)
    font = ImageFont.truetype("arial.ttf", size=10)  # Adjust font size if needed

    # Create drawing context for text rendering
    draw = ImageDraw.Draw(blank_space)
    
    # Add units to the metrics text, including the correct square symbol for area (²)
    metrics_text = (
        f"Length: {length:.2f} mm | Width: {width:.2f} mm | Area: {area:.2f} mm² | "
        f"Perimeter: {perimeter:.2f} mm | Circularity: {circularity:.2f}"
    )
    
    # Calculate text size to center it
    text_bbox = draw.textbbox((0, 0), metrics_text, font=font)
    text_width = text_bbox[2] - text_bbox[0]  # Width of the text
    text_height = text_bbox[3] - text_bbox[1]  # Height of the text
    
    # Calculate the center position
    position_x = (blank_space.width - text_width) // 2
    position_y = padding_height // 2  # Position text with some space above

    # Render the text onto the blank space using Pillow
    draw.text((position_x, position_y), metrics_text, font=font, fill=(255, 255, 255))

    # Convert blank_space back to OpenCV format
    blank_space_cv = cv2.cvtColor(np.array(blank_space), cv2.COLOR_RGB2BGR)

    # Stack comparison image and metrics with the fixed black space below
    comparison_with_metrics = np.vstack((comparison_image, blank_space_cv))

    # Save final comparison image with metrics
    cv2.imwrite(COMPARISON_IMAGE_PATH, comparison_with_metrics)
    
    print(f"Predicted mask saved at {PREDICTED_MASK_PATH}")
    print(f"Comparison image saved at {COMPARISON_IMAGE_PATH}")
    print(f"Color-coded comparison image saved at {COLORED_COMPARISON_PATH}")



# Run the function to process the single image
process_single_image()

print("Single image processed successfully with metrics.")
