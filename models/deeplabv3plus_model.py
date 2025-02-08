import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import ResNet50

def build_deeplabv3plus(input_shape=(256, 256, 3), num_classes=1):
    base_model = tf.keras.applications.ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)

    # Atrous Spatial Pyramid Pooling (ASPP)
    layer_names = [
        "conv4_block6_2_relu",  # 16x16 output size
        "conv4_block3_2_relu",  # 32x32 output size
        "conv3_block4_2_relu"   # 64x64 output size
    ]
    layer_outputs = [base_model.get_layer(name).output for name in layer_names]

    for layer in base_model.layers:
        layer.trainable = False

    # ASPP Block
    x = layers.Conv2D(256, 3, padding="same", dilation_rate=6, activation="relu")(layer_outputs[-1])
    x = layers.BatchNormalization()(x)

    x = layers.Conv2D(256, 3, padding="same", dilation_rate=12, activation="relu")(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv2D(256, 3, padding="same", dilation_rate=18, activation="relu")(x)
    x = layers.BatchNormalization()(x)

    # Bilinear Upsampling to 256x256
    x = layers.UpSampling2D(size=(4, 4), interpolation="bilinear")(x)  # Upsample from 64x64 -> 256x256
    x = layers.UpSampling2D(size=(2, 2), interpolation="bilinear")(x)  # Upsample from 64x64 -> 256x256

    # Final output layer
    x = layers.Conv2D(num_classes, 1, activation="sigmoid")(x)  # Final upsampled mask, 256x256
    
    return Model(inputs=base_model.input, outputs=x, name="DeepLabV3Plus")

if __name__ == "__main__":
    input_shape = (256, 256, 3)
    model = build_deeplabv3plus(input_shape=input_shape)
    model.summary()
