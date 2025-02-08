import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from models.deeplabv3plus_model import build_deeplabv3plus

# Paths
TRAIN_IMAGES_PATH = 'data/images/train/original'
TRAIN_MASKS_PATH = 'data/images/train/masks'
OUTPUT_DIR_PREDICTED_MASKS = 'outputs/deeplabv3plus/training_predicted_masks'
OUTPUT_DIR_COMPARISON = 'outputs/deeplabv3plus/training_comparison'

# Create directories if they don't exist
os.makedirs(OUTPUT_DIR_PREDICTED_MASKS, exist_ok=True)
os.makedirs(OUTPUT_DIR_COMPARISON, exist_ok=True)

# Constants
IMAGE_SIZE = (256, 256)
BATCH_SIZE = 8
EPOCHS = 5

# Load and preprocess the images and masks
def load_images_and_masks():
    images = []
    masks = []
    image_names = sorted(os.listdir(TRAIN_IMAGES_PATH))
    
    for image_name in image_names:
        # Load image
        image_path = os.path.join(TRAIN_IMAGES_PATH, image_name)
        image = cv2.imread(image_path)
        image = cv2.resize(image, IMAGE_SIZE)
        images.append(image)
        
        # Load corresponding mask
        mask_path = os.path.join(TRAIN_MASKS_PATH, image_name)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, IMAGE_SIZE)
        masks.append(mask)
    
    images = np.array(images) / 255.0  # Normalize images
    masks = np.array(masks) / 255.0    # Normalize masks (0 or 1)
    masks = np.expand_dims(masks, axis=-1)  # Add channel dimension
    
    return np.array(images), np.array(masks)

x_data, y_data = load_images_and_masks()

# Split data into training and validation sets
x_train, x_val, y_train, y_val = train_test_split(x_data, y_data, test_size=0.2, random_state=42)

# Build the DeepLabV3+ model
model = build_deeplabv3plus(input_shape=(256, 256, 3), num_classes=1)

# Compile the model
model.compile(optimizer=Adam(learning_rate=1e-4), loss="binary_crossentropy", metrics=["accuracy"])

# Train the model
history = model.fit(x_train, y_train, validation_data=(x_val, y_val), batch_size=BATCH_SIZE, epochs=EPOCHS)

# Generate predictions and save output images
for i in range(len(x_val)):
    original_image = x_val[i]
    ground_truth_mask = y_val[i]
    
    # Predict mask
    pred_mask = model.predict(np.expand_dims(original_image, axis=0))[0]
    pred_mask = (pred_mask > 0.5).astype(np.uint8)  # Binarize the prediction (thresholding)
    
    # Save predicted mask
    pred_mask_path = os.path.join(OUTPUT_DIR_PREDICTED_MASKS, f'predicted_mask_{i}.png')
    cv2.imwrite(pred_mask_path, pred_mask * 255)  # Convert to binary image (0 or 255)
    
    # Convert grayscale masks to RGB for comparison
    ground_truth_mask_rgb = np.repeat(ground_truth_mask, 3, axis=-1)  # Convert (256, 256, 1) to (256, 256, 3)
    pred_mask_rgb = np.repeat(pred_mask, 3, axis=-1)  # Convert (256, 256, 1) to (256, 256, 3)
    
    # Create comparison image (original, ground truth, predicted)
    comparison_image = np.hstack((original_image, ground_truth_mask_rgb, pred_mask_rgb))
    
    # Save comparison image
    comparison_image_path = os.path.join(OUTPUT_DIR_COMPARISON, f'comparison_{i}.png')
    cv2.imwrite(comparison_image_path, comparison_image * 255)

print("Training and output generation complete!")
