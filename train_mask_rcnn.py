import numpy as np
import os
import cv2
import tensorflow as tf
from models.mask_rcnn_model import build_mask_rcnn
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Directories
IMAGE_DIR = 'data/images/train/original'
MASK_DIR = 'data/images/train/masks'
OUTPUT_DIR = 'outputs/mask_rcnn/training_predicted_masks'
COMPARISON_OUTPUT_DIR = 'outputs/mask_rcnn/training_comparison'

# Hyperparameters
BATCH_SIZE = 8
EPOCHS = 5
IMG_HEIGHT, IMG_WIDTH = 256, 256

# Load dataset
def load_images_and_masks():
    images = []
    masks = []
    
    for filename in os.listdir(IMAGE_DIR):
        image_path = os.path.join(IMAGE_DIR, filename)
        mask_path = os.path.join(MASK_DIR, filename)
        
        if os.path.exists(image_path) and os.path.exists(mask_path):
            # Load and resize the image
            image = cv2.imread(image_path)
            image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
            images.append(image)
            
            # Load and resize the mask
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, (IMG_WIDTH, IMG_HEIGHT))
            mask = np.expand_dims(mask, axis=-1)  # Add channel dimension (shape: 256x256x1)
            masks.append(mask)
    
    return np.array(images), np.array(masks)

x, y = load_images_and_masks()
x = x / 255.0  # Normalize images
y = y / 255.0  # Normalize masks

# Train-test split
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)

# Build and compile model
model = build_mask_rcnn(input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Train the model
history = model.fit(x_train, y_train, validation_data=(x_val, y_val), batch_size=BATCH_SIZE, epochs=EPOCHS)

# Create directories if they don't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(COMPARISON_OUTPUT_DIR, exist_ok=True)

# Generate and save predicted masks and comparison images
for i in range(len(x_val)):
    original_image = x_val[i]
    ground_truth_mask = y_val[i]
    
    # Predict mask using the model
    pred_mask = model.predict(np.expand_dims(original_image, axis=0))[0]
    pred_mask = (pred_mask > 0.5).astype(np.uint8)  # Threshold prediction

    # Convert ground truth and predicted masks to RGB
    ground_truth_mask_rgb = np.repeat(ground_truth_mask, 3, axis=-1)  # Convert 256x256x1 -> 256x256x3
    pred_mask_rgb = np.repeat(pred_mask, 3, axis=-1)  # Convert 256x256x1 -> 256x256x3

    # Save the predicted mask as a grayscale image
    mask_output_path = os.path.join(OUTPUT_DIR, f'predicted_mask_{i+1}.png')
    cv2.imwrite(mask_output_path, pred_mask * 255)

    # Create the comparison image by combining original image, ground truth mask, and predicted mask
    comparison_image = np.hstack((original_image, ground_truth_mask_rgb, pred_mask_rgb))

    # Save the comparison image
    comparison_output_path = os.path.join(COMPARISON_OUTPUT_DIR, f'comparison_{i+1}.png')
    cv2.imwrite(comparison_output_path, comparison_image * 255)

    print(f"Saved predicted mask and comparison for image {i+1}/{len(x_val)}")

print("Training and output generation complete!")
