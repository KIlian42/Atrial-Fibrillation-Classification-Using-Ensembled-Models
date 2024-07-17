from tensorflow import keras
import numpy as np

def conv(i, filters=16, kernel_size=3, strides=1):
    i = keras.layers.Conv1D(filters=filters, kernel_size=kernel_size, strides=strides, padding="same")(i)
    i = keras.layers.BatchNormalization()(i)
    i = keras.layers.LeakyReLU()(i)
    i = keras.layers.SpatialDropout1D(0.1)(i)
    return i

def inception(inputs, filters):
    conv1 = conv(inputs, filters, kernel_size=3)
    conv2 = conv(inputs, filters, kernel_size=5)
    conv3 = conv(inputs, filters, kernel_size=7)
    conv4 = conv(inputs, filters, kernel_size=9)
    conv5 = conv(inputs, filters, kernel_size=11)
    conv6 = conv(inputs, filters, kernel_size=13)
    conv7 = conv(inputs, filters, kernel_size=15)
    concatenated = keras.layers.Concatenate()([conv1, conv2, conv3, conv4, conv5, conv6, conv7])
    return concatenated

def residual_unit(x, filters, layers=1):
    inp = x
    for i in range(layers):
        x = inception(x, filters)
        # x = conv(x, filters)
    return keras.layers.add([x, inp])

def conv_block(x, filters, strides):
    x = conv(x, filters*7)
    x = residual_unit(x, filters)
    if strides > 1:
        x = keras.layers.AveragePooling1D(strides, strides)(x)
    return x

def build_model(input_shape, num_classes):
    inp = keras.layers.Input(input_shape)
    x = inp
    x = conv_block(x, 16, 2)
    x = conv_block(x, 32, 2)
    # x = conv_block(x, 64, 2)
    # x = conv_block(x, 128, 2)
    x = keras.layers.GlobalAveragePooling1D()(x)
    x = keras.layers.Dense(num_classes, activation="sigmoid")(x)
    model = keras.models.Model(inp, x)
    return model

# ==================
num_classes = 5
x_train = np.random.rand(20000, 2000, 1)
input_shape = x_train.shape[1:]
model = build_model(input_shape=input_shape, num_classes=num_classes)
model.summary()