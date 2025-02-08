import tensorflow as tf
from tensorflow.keras import layers, Model

def build_unet(input_shape=(256, 256, 3), num_classes=1):
    inputs = tf.keras.Input(input_shape)

    # Encoder
    conv1 = layers.Conv2D(64, 3, activation='relu', padding='same')(inputs)
    conv1 = layers.Conv2D(64, 3, activation='relu', padding='same')(conv1)
    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = layers.Conv2D(128, 3, activation='relu', padding='same')(pool1)
    conv2 = layers.Conv2D(128, 3, activation='relu', padding='same')(conv2)
    pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = layers.Conv2D(256, 3, activation='relu', padding='same')(pool2)
    conv3 = layers.Conv2D(256, 3, activation='relu', padding='same')(conv3)
    pool3 = layers.MaxPooling2D(pool_size=(2, 2))(conv3)

    # Bottleneck
    conv4 = layers.Conv2D(512, 3, activation='relu', padding='same')(pool3)
    conv4 = layers.Conv2D(512, 3, activation='relu', padding='same')(conv4)

    # Decoder
    upconv5 = layers.Conv2DTranspose(256, 2, strides=(2, 2), padding='same')(conv4)
    upconv5 = layers.concatenate([upconv5, conv3])
    conv5 = layers.Conv2D(256, 3, activation='relu', padding='same')(upconv5)
    conv5 = layers.Conv2D(256, 3, activation='relu', padding='same')(conv5)

    upconv6 = layers.Conv2DTranspose(128, 2, strides=(2, 2), padding='same')(conv5)
    upconv6 = layers.concatenate([upconv6, conv2])
    conv6 = layers.Conv2D(128, 3, activation='relu', padding='same')(upconv6)
    conv6 = layers.Conv2D(128, 3, activation='relu', padding='same')(conv6)

    upconv7 = layers.Conv2DTranspose(64, 2, strides=(2, 2), padding='same')(conv6)
    upconv7 = layers.concatenate([upconv7, conv1])
    conv7 = layers.Conv2D(64, 3, activation='relu', padding='same')(upconv7)
    conv7 = layers.Conv2D(64, 3, activation='relu', padding='same')(conv7)

    # Output layer
    outputs = layers.Conv2D(num_classes, 1, activation='sigmoid')(conv7)

    return Model(inputs=[inputs], outputs=[outputs])

if __name__ == "__main__":
    model = build_unet(input_shape=(256, 256, 3))
    model.summary()
