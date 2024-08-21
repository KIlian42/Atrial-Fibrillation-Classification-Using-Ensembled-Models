import keras
import tensorflow as tf
from tensorflow.keras import regularizers

def transformer_encoder(input, input_shape, num_heads, key_dim, ff_dim, dropout):
    # Multi-Head Attention
    x = keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim, dropout=dropout, kernel_regularizer=regularizers.l2(0.001))(input, input)
    # Add & Normalize
    res = x + input
    x = keras.layers.LayerNormalization(epsilon=1e-6)(res)
    # Feed-Forward Layer
    x = keras.layers.Flatten(input_shape=input_shape)(x)
    x = keras.layers.Dense(units=ff_dim, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
    x = keras.layers.Dense(input_shape[0] * input_shape[1], kernel_regularizer=regularizers.l2(0.001))(x)
    x = keras.layers.Reshape(input_shape)(x)
    x = keras.layers.Dropout(rate=dropout)(x)
    # Add & Normalize
    x = x + res
    x = keras.layers.LayerNormalization(epsilon=1e-6)(x)
    return x

def conv(i, filters=16, kernel_size=5, strides=1):
    i = keras.layers.Conv1D(filters=filters, kernel_size=kernel_size, strides=strides, padding="same", dilation_rate=2)(i) # dilation_rate=2
    i = keras.layers.BatchNormalization()(i)
    i = keras.layers.LeakyReLU()(i)
    i = keras.layers.SpatialDropout1D(0.1)(i)
    return i

def inception(inputs, filters):
    conv1 = conv(inputs, filters, kernel_size=3)
    conv2 = conv(inputs, filters, kernel_size=5)
    # conv3 = conv(inputs, filters, kernel_size=7)
    conv4 = conv(inputs, filters, kernel_size=9)
    # conv5 = conv(inputs, filters, kernel_size=11)
    # conv6 = conv(inputs, filters, kernel_size=13)
    conv7 = conv(inputs, filters, kernel_size=15)
    # concatenated = keras.layers.Concatenate()([conv1, conv2, conv3, conv4, conv5, conv6, conv7])
    concatenated = keras.layers.Concatenate()([conv1, conv2, conv4, conv7])
    return concatenated

def residual_unit(x, filters, layers=1):
    inp = x
    for i in range(layers):
        x = inception(x, filters)
        # x = conv(x, filters)
    return keras.layers.add([x, inp])

def conv_block(x, filters, strides):
    x = conv(x, filters*4)
    # x = conv(x, filters)
    x = residual_unit(x, filters)
    if strides > 1:
        x = keras.layers.AveragePooling1D(strides, strides)(x)
    return x

def build_model(input_shape, num_classes):
    inp = keras.layers.Input(shape=input_shape)
    outputs = []
    for i in range(input_shape[0]):
        x = inp[:, i, :, :]  # Select the i-th sequence (shape: (batch, 2000, 1))
        x = conv_block(x, 16, 2)
        x = conv_block(x, 32, 2)
        x = conv_block(x, 64, 2)
        x = keras.layers.GlobalAveragePooling1D()(x)
        outputs.append(x)

    outputs = keras.layers.Lambda(lambda x: tf.stack(x, axis=1))(outputs)

    x = transformer_encoder(outputs, (12, 256), 8, 32, 24, 0.1)
    x = transformer_encoder(x, (12, 256), 8, 32, 24, 0.1)
    x = transformer_encoder(x, (12, 256), 8, 32, 24, 0.1)

    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(num_classes, activation="sigmoid")(x)
    model = keras.models.Model(inp, x)
    return model
    
    
    # Final dense layer
    x = keras.layers.Dense(num_classes, activation="sigmoid")(x)
    
    model = keras.models.Model(inp, x)
    return model