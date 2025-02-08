import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from models.unet_model import build_unet
import cv2

# Constants
BATCH_SIZE = 8
EPOCHS = 50
IMAGE_SIZE = (256, 256)

# Paths
train_images_path = "data/images/train/original"
train_masks_path = "data/images/train/masks"
predicted_masks_path = "outputs/unet/training_predicted_masks"
comparison_images_path = "outputs/unet/training_comparison"

# Create output directories if they don't exist
os.makedirs(predicted_masks_path, exist_ok=True)
os.makedirs(comparison_images_path, exist_ok=True)

# Load images and masks
def load_data(image_dir, mask_dir):
    image_filenames = sorted(os.listdir(image_dir))
    images = []
    masks = []
    
    for image_filename in image_filenames:
        image = load_img(os.path.join(image_dir, image_filename), target_size=IMAGE_SIZE)
        mask = load_img(os.path.join(mask_dir, image_filename), target_size=IMAGE_SIZE, color_mode="grayscale")
        
        image = img_to_array(image) / 255.0
        mask = img_to_array(mask) / 255.0
        
        images.append(image)
        masks.append(mask)
    
    return np.array(images), np.array(masks)

x_train, y_train = load_data(train_images_path, train_masks_path)

# Split the data (you can adjust the split ratio if needed)
split_index = int(0.8 * len(x_train))
x_val = x_train[split_index:]
y_val = y_train[split_index:]
x_train = x_train[:split_index]
y_train = y_train[:split_index]

# Build U-Net model
model = build_unet(input_shape=(256, 256, 3))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train model
history = model.fit(x_train, y_train, validation_data=(x_val, y_val), batch_size=BATCH_SIZE, epochs=EPOCHS)

# Save model after training
model.save('outputs/unet/unet_model.h5')

# Generate predictions and save results
for i in range(len(x_val)):
    original_image = x_val[i]
    ground_truth_mask = y_val[i]
    
    # Predict mask
    pred_mask = model.predict(np.expand_dims(original_image, axis=0))[0]
    
    # Convert to binary mask
    pred_mask = (pred_mask > 0.5).astype(np.uint8)
    
    # Save predicted mask
    pred_mask_path = os.path.join(predicted_masks_path, f"pred_mask_{i}.png")
    cv2.imwrite(pred_mask_path, pred_mask * 255)
    
    # Create comparison image
    original_image_rgb = (original_image * 255).astype(np.uint8)
    ground_truth_mask_rgb = np.repeat((ground_truth_mask * 255).astype(np.uint8), 3, axis=-1)
    pred_mask_rgb = np.repeat((pred_mask * 255).astype(np.uint8), 3, axis=-1)
    
    comparison_image = np.hstack((original_image_rgb, ground_truth_mask_rgb, pred_mask_rgb))
    
    # Save comparison image
    comparison_image_path = os.path.join(comparison_images_path, f"comparison_{i}.png")
    cv2.imwrite(comparison_image_path, comparison_image)

print("Training and saving complete!")


# import os
# import numpy as np
# import tensorflow as tf
# from tensorflow.keras import layers, Model
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
# from sklearn.model_selection import train_test_split
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from models.unet_model import build_unet  # Ensure this function is correctly defined in unet_model.py

# # Set parameters
# input_shape = (256, 256, 3)
# num_classes = 1
# BATCH_SIZE = 16
# EPOCHS = 50  # Increase epochs
# DATA_DIR = "data/images/train/original/"
# MASK_DIR = "data/images/train/masks/"
# OUTPUT_DIR = "outputs/unet/"
# PREDICTED_MASKS_DIR = os.path.join(OUTPUT_DIR, "training_predicted_masks/")
# COMPARISON_DIR = os.path.join(OUTPUT_DIR, "training_comparison/")

# # Clear existing output directories
# os.makedirs(PREDICTED_MASKS_DIR, exist_ok=True)
# os.makedirs(COMPARISON_DIR, exist_ok=True)

# # Load images and masks
# def load_data(data_dir, mask_dir):
#     images = []
#     masks = []
    
#     for image_name in os.listdir(data_dir):
#         image_path = os.path.join(data_dir, image_name)
#         mask_path = os.path.join(mask_dir, image_name)

#         if image_name.endswith('.png') and os.path.exists(mask_path):
#             img = tf.keras.preprocessing.image.load_img(image_path, target_size=input_shape[:2])
#             img = tf.keras.preprocessing.image.img_to_array(img) / 255.0
            
#             mask = tf.keras.preprocessing.image.load_img(mask_path, target_size=input_shape[:2], color_mode='grayscale')
#             mask = tf.keras.preprocessing.image.img_to_array(mask) / 255.0
#             mask = np.expand_dims(mask, axis=-1)  # Add channel dimension

#             images.append(img)
#             masks.append(mask)

#     return np.array(images), np.array(masks)

# x, y = load_data(DATA_DIR, MASK_DIR)

# # Split the dataset into training and validation sets
# x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)

# # Data augmentation
# datagen = ImageDataGenerator(
#     rotation_range=15,
#     width_shift_range=0.1,
#     height_shift_range=0.1,
#     shear_range=0.01,
#     zoom_range=[0.9, 1.25],
#     horizontal_flip=True,
#     fill_mode='reflect'
# )

# # Build U-Net model
# model = build_unet(input_shape=input_shape, num_classes=num_classes)

# # Compile the model with a reduced learning rate
# optimizer = Adam(learning_rate=1e-4)
# model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# # Callbacks for early stopping and model checkpointing
# early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
# checkpoint = ModelCheckpoint(os.path.join(OUTPUT_DIR, 'unet_model.keras'), 
#                              monitor='val_loss', 
#                              save_best_only=True,
#                              mode='min', 
#                              verbose=1)

# # Fit the model using the data generator
# history = model.fit(datagen.flow(x_train, y_train, batch_size=BATCH_SIZE),
#                     validation_data=(x_val, y_val),
#                     epochs=EPOCHS,
#                     callbacks=[early_stopping, checkpoint])

# # Generate predicted masks and comparison images
# for i in range(len(x_val)):
#     original_image = x_val[i]
#     ground_truth_mask = y_val[i]
    
#     # Predict mask
#     pred_mask = model.predict(np.expand_dims(original_image, axis=0))[0]
    
#     # Convert masks to RGB for comparison
#     pred_mask_rgb = np.zeros((*pred_mask.shape[:2], 3))
#     pred_mask_rgb[..., 0] = pred_mask[..., 0]  # Red channel
#     pred_mask_rgb[..., 1] = pred_mask[..., 0]  # Green channel
#     pred_mask_rgb[..., 2] = pred_mask[..., 0]  # Blue channel
    
#     # Save the predicted mask and comparison image
#     pred_mask_rgb = (pred_mask_rgb * 255).astype(np.uint8)
#     comparison_image = np.hstack((original_image, ground_truth_mask.squeeze(), pred_mask_rgb))
    
#     # Save the images
#     tf.keras.preprocessing.image.save_img(os.path.join(PREDICTED_MASKS_DIR, f"pred_mask_{i}.png"), pred_mask_rgb)
#     tf.keras.preprocessing.image.save_img(os.path.join(COMPARISON_DIR, f"comparison_{i}.png"), comparison_image)

# print("Training completed and outputs saved.")
