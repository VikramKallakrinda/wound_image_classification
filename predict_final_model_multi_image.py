import os
import zipfile
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from calculate_metrics import calculate_metrics  # Import the metrics calculation function
from PIL import Image, ImageDraw, ImageFont  # Import Pillow for advanced text rendering

# Constants
IMAGE_SIZE = (256, 256)
OUTPUT_DIR = 'outputs/final/Results'

# Load the trained model (replace 'model_path.h5' with your actual model path)
model = load_model('outputs/unet/unet_model.h5', compile=False)

def generate_outputs(image_path):
    # Load and resize the original image
    original_image = Image.open(image_path).convert('RGB')  # Ensure the image is in RGB format
    original_image_resized = original_image.resize(IMAGE_SIZE)
    original_image_array = img_to_array(original_image_resized) / 255.0

    # Predict mask
    pred_mask = model.predict(np.expand_dims(original_image_array, axis=0))[0]
    pred_mask = (pred_mask > 0.5).astype(np.uint8)  # Binarize mask

    # Save outputs
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    
    # Save predicted mask
    output_mask_path = os.path.join(OUTPUT_DIR, f'{base_name}_predicted_mask.png')
    pred_mask_resized = cv2.resize(pred_mask, (original_image.size[0], original_image.size[1]))  # Resize to original size
    cv2.imwrite(output_mask_path, pred_mask_resized * 255)

    # Convert the original image from RGB to BGR (since OpenCV uses BGR format)
    original_image_bgr = cv2.cvtColor(np.array(original_image), cv2.COLOR_RGB2BGR)
    
    # Create predicted mask in 3 channels (grayscale to RGB-like BGR for OpenCV)
    pred_mask_bgr = np.repeat(pred_mask_resized[:, :, np.newaxis], 3, axis=2) * 255

    # ------------------- Color-coded overlay -------------------
    # Create translucent overlays
    red_overlay = np.zeros_like(original_image_bgr, dtype=np.uint8)
    blue_overlay = np.zeros_like(original_image_bgr, dtype=np.uint8)

    # Set colors for overlays
    red_overlay[:, :, 2] = 255  # Red channel to max (BGR format)
    blue_overlay[:, :, 0] = 255  # Blue channel to max (BGR format)

    # Create the translucent overlays based on the predicted mask
    alpha_red = 0.5  # Transparency factor for red overlay
    alpha_blue = 0.5  # Transparency factor for blue overlay
    color_overlay = np.where(pred_mask_resized[:, :, np.newaxis], 
                             cv2.addWeighted(original_image_bgr, 1, red_overlay, alpha_red, 0),
                             cv2.addWeighted(original_image_bgr, 1, blue_overlay, alpha_blue, 0))

    # Save the color-coded overlay image
    output_overlay_path = os.path.join(OUTPUT_DIR, f'{base_name}_color_overlay.png')
    cv2.imwrite(output_overlay_path, color_overlay)

    # Calculate metrics
    length, width, area, perimeter, circularity = calculate_metrics(pred_mask_resized)

    # Create comparison image (original + predicted mask + color overlay side by side)
    comparison_image = np.hstack((original_image_bgr, pred_mask_bgr, color_overlay))

    # Create a blank space for writing metrics below the image
    padding_height = 100
    blank_space_height = 50  # Height for the blank space
    blank_space = Image.new("RGB", (comparison_image.shape[1], blank_space_height), (0, 0, 0))
    
    # Load a font (adjust the path and size as needed)
    font_path = "arial.ttf"  # Ensure the font file is accessible
    font = ImageFont.truetype(font_path, size=10)

    # Create drawing context for text rendering
    draw = ImageDraw.Draw(blank_space)
    
    # Prepare metrics text
    metrics_text = (
        f"Length: {length:.2f} mm | Width: {width:.2f} mm | Area: {area:.2f} mm² | "
        f"Perimeter: {perimeter:.2f} mm | Circularity: {circularity:.2f}"
    )

    # Calculate text size to center it
    text_bbox = draw.textbbox((0, 0), metrics_text, font=font)
    text_width = text_bbox[2] - text_bbox[0]  # Width of the text
    position_x = (blank_space.width - text_width) // 2  # Centered position

     # Calculate the center position
    position_x = (blank_space.width - text_width) // 2
    position_y = padding_height // 2  # Position text with some space above


    # Render the metrics text onto the blank space
    draw.text((position_x, 10), metrics_text, font=font, fill=(255, 255, 255))

    # Stack comparison image and metrics
    comparison_with_metrics = np.vstack((comparison_image, np.array(blank_space)))

    # Save final comparison image with metrics
    output_comparison_path = os.path.join(OUTPUT_DIR, f'{base_name}_comparison.png')
    cv2.imwrite(output_comparison_path, comparison_with_metrics)

    return output_mask_path, output_overlay_path, output_comparison_path

def process_images(input_images):
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    for image_path in input_images:
        print(f'Processing {image_path}...')
        generate_outputs(image_path)

def process_single_image(image_path):
    return generate_outputs(image_path)

def process_multiple_images(image_paths):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, 'predicted_masks'), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, 'color_overlays'), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, 'comparison_images'), exist_ok=True)

    for image_path in image_paths:
        print(f'Processing {image_path}...')
        output_mask_path, output_overlay_path, output_comparison_path = generate_outputs(image_path)

        # Move files to respective folders
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        os.rename(output_mask_path, os.path.join(OUTPUT_DIR, 'predicted_masks', f'{base_name}_predicted_mask.png'))
        os.rename(output_overlay_path, os.path.join(OUTPUT_DIR, 'color_overlays', f'{base_name}_color_overlay.png'))
        os.rename(output_comparison_path, os.path.join(OUTPUT_DIR, 'comparison_images', f'{base_name}_comparison.png'))

    # Create a zip file of the results
    with zipfile.ZipFile('outputs/final/results.zip', 'w') as zipf:
        for folder_name in ['predicted_masks', 'color_overlays', 'comparison_images']:
            folder_path = os.path.join(OUTPUT_DIR, folder_name)
            for file_name in os.listdir(folder_path):
                zipf.write(os.path.join(folder_path, file_name), os.path.join(folder_name, file_name))

    print("All results processed and zipped successfully.")

if __name__ == "__main__":
    choice = input("Choose an option:\n1. Single Image\n2. Multiple Images\nEnter 1 or 2: ")

    if choice == '1':
        image_path = input("Enter the path of the image: ")
        process_single_image(image_path)

    elif choice == '2':
        image_folder = input("Enter the folder path containing images: ")
        input_images = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith(('png', 'jpg', 'jpeg'))]
        process_multiple_images(input_images)

    else:
        print("Invalid choice. Please choose 1 or 2.")
