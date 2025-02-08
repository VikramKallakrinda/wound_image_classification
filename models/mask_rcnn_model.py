import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import ResNet50

def build_mask_rcnn(input_shape=(256, 256, 3), num_classes=1):
    # Backbone model (ResNet50)
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    
    # Extract feature maps from different layers
    layer_names = [
        "conv4_block6_2_relu",  # 32x32 output
        "conv3_block4_2_relu",  # 64x64 output
        "conv2_block3_2_relu",  # 128x128 output
        "conv1_relu"            # 256x256 output
    ]
    layer_outputs = [base_model.get_layer(name).output for name in layer_names]
    
    # Use the deepest feature map (32x32)
    x = layer_outputs[0]
    
    # Upsample step 1: 32x32 -> 64x64
    x = layers.Conv2DTranspose(256, 3, strides=2, padding="same")(x)  # 64x64
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    # Upsample step 2: 64x64 -> 128x128
    x = layers.Conv2DTranspose(128, 3, strides=2, padding="same")(x)  # 128x128
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    # Upsample step 3: 128x128 -> 256x256
    x = layers.Conv2DTranspose(64, 3, strides=2, padding="same")(x)  # 256x256
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    # Upsample step 3: 128x128 -> 256x256
    x = layers.Conv2DTranspose(64, 3, strides=2, padding="same")(x)  # 256x256
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    # Output layer (single channel for mask prediction)
    outputs = layers.Conv2D(num_classes, 1, activation="sigmoid")(x)  # Shape: (256x256x1)

    return Model(inputs=base_model.input, outputs=outputs, name="MaskRCNN")

if __name__ == "__main__":
    model = build_mask_rcnn()
    model.summary()
