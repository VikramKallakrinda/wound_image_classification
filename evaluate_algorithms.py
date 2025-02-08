import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.preprocessing.image import load_img, img_to_array, save_img
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt
from models.unet_model import create_unet_model

def preprocess_image(image_path, target_size=(256, 256)):
    image = load_img(image_path, color_mode='grayscale')
    image = img_to_array(image)
    image = tf.image.resize(image, target_size)
    image = image / 255.0  # Normalize the image
    return image

def preprocess_mask(mask_path, target_size=(256, 256)):
    mask = load_img(mask_path, color_mode='grayscale')
    mask = img_to_array(mask)
    mask = tf.image.resize(mask, target_size)
    mask = mask / 255.0  # Normalize the mask
    return mask

def load_data(image_dir, mask_dir, target_size=(256, 256)):
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])
    mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith('.png')])
    
    images = np.array([preprocess_image(os.path.join(image_dir, f), target_size) for f in image_files])
    masks = np.array([preprocess_mask(os.path.join(mask_dir, f), target_size) for f in mask_files])
    
    return images, masks

def save_comparison_images(original_image, manual_mask, predicted_mask, save_path):
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.title('Original')
    plt.imshow(original_image.squeeze(), cmap='gray')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.title('Manual Mask')
    plt.imshow(manual_mask.squeeze(), cmap='gray')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.title('Predicted Mask')
    plt.imshow(predicted_mask.squeeze(), cmap='gray')
    plt.axis('off')
    
    plt.savefig(save_path)
    plt.close()

if __name__ == "__main__":
    image_dir = 'data/images/train/original'
    mask_dir = 'data/images/train/masks'
    output_predicted_masks_dir = 'outputs/training/training_predicted_masks'
    output_comparison_dir = 'outputs/training/training_comparision'
    
    target_size = (256, 256)  # Adjust as needed
    images, masks = load_data(image_dir, mask_dir, target_size)
    
    model = create_unet_model(input_shape=(256, 256, 1))
    model_checkpoint = ModelCheckpoint('unet_model.keras', save_best_only=True)
    early_stopping = EarlyStopping(patience=5)
    
    # Training the model
    model.fit(images, masks, batch_size=16, epochs=10, validation_split=0.1, callbacks=[model_checkpoint, early_stopping])
    
    # Save predicted masks
    os.makedirs(output_predicted_masks_dir, exist_ok=True)
    for i in range(len(images)):
        img = images[i:i+1]
        manual_mask = masks[i:i+1]
        predicted_mask = model.predict(img)
        predicted_mask = (predicted_mask > 0.5).astype(np.uint8)  # Binarize the mask
        
        original_image = load_img(os.path.join(image_dir, sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])[i]), color_mode='grayscale')
        original_image = img_to_array(original_image)
        
        save_img(os.path.join(output_predicted_masks_dir, f'{i}_mask.png'), predicted_mask[0])
        
        save_comparison_images(original_image, manual_mask[0], predicted_mask[0],
                               os.path.join(output_comparison_dir, f'comparison_{i}.png'))
