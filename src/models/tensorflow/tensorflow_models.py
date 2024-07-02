import numpy as np
import tensorflow as tf
from keras.layers import (
    Input,
    MultiHeadAttention,
    Conv1D,
    Conv2D,
    BatchNormalization,
    SpatialDropout1D,
    SpatialDropout2D,
    AveragePooling1D,
    AveragePooling2D,
    GlobalAveragePooling1D,
    Reshape,
    Flatten,
    Dense,
    Masking,
    Lambda,
    LayerNormalization,
    Add,
    Dropout,
    GRU,
)
from keras.models import Model
from keras.activations import sigmoid, relu, leaky_relu, gelu
import keras.backend as K


# 1-lead CNN + Transformer
def build_model_CNN_Transformer_1lead_simple(input_shape, num_classes):
    input_layer = Input(input_shape)
    # Encoder block/Attention mechanisms
    i = MultiHeadAttention(num_heads=8, key_dim=50, dropout=0.3)(
        input_layer, input_layer
    )
    # Flatten
    i = Flatten()(i)
    # Feedforward Softmax
    i = Dense(num_classes, activation="softmax")(i)
    return Model(inputs=input_layer, outputs=i)


# 1-lead CNN + Transformer
def build_model_CNN_Transformer_1lead(input_shape, num_classes):
    input_layer = Input(input_shape)
    # Masking for padded/truncated data
    i = Masking(mask_value=0)(input_layer)
    # Conv1
    i = Conv1D(filters=16, kernel_size=9, strides=1, padding="same")(i)
    i = BatchNormalization()(i)
    i = gelu(i)
    i = SpatialDropout1D(0.1)(i)
    # i = AveragePooling1D(2)(i)
    # Conv2
    i = Conv1D(filters=32, kernel_size=9, strides=1, padding="same")(i)
    i = BatchNormalization()(i)
    i = gelu(i)
    i = SpatialDropout1D(0.1)(i)
    # i = AveragePooling1D(2)(i)
    # Conv3
    i = Conv1D(filters=64, kernel_size=9, strides=1, padding="same")(i)
    i = BatchNormalization()(i)
    i = gelu(i)
    i = SpatialDropout1D(0.1)(i)
    # i = AveragePooling1D(2)(i)
    # Channel Average Pooling and Reshaping
    i = GlobalAveragePooling1D(data_format="channels_first")(i)
    # i = keras.layers.Reshape((250, 1))(i)
    i = Reshape((2000, 1))(i)
    # Encoder block/Attention mechanisms
    i = MultiHeadAttention(num_heads=8, key_dim=50, dropout=0.3)(i, i)
    # Flatten
    i = Flatten()(i)
    # Feedforward Softmax
    i = Dense(num_classes, activation="softmax")(i)
    return Model(inputs=input_layer, outputs=i)


# 12-lead CNN + Transformer
def build_model_CNN_Transformer_12lead(input_shape, num_classes):
    input_layer = Input(input_shape)
    # Masking for padded/truncated data
    i = Masking(mask_value=0)(input_layer)

    # Conv1
    i = Conv2D(filters=16, kernel_size=9, strides=1, padding="same")(i)
    i = BatchNormalization()(i)
    i = relu()(i)
    i = SpatialDropout2D(0.1)(i)
    i = AveragePooling2D(pool_size=(1, 2))(i)

    # Conv2
    i = Conv2D(filters=32, kernel_size=9, strides=1, padding="same")(i)
    i = BatchNormalization()(i)
    i = relu()(i)
    i = SpatialDropout2D(0.1)(i)
    i = AveragePooling2D(pool_size=(1, 2))(i)

    # Conv3
    i = Conv2D(filters=64, kernel_size=9, strides=1, padding="same")(i)
    i = BatchNormalization()(i)
    i = relu()(i)
    i = SpatialDropout2D(0.1)(i)
    i = AveragePooling2D(pool_size=(1, 2))(i)

    # Conv4
    i = Conv2D(filters=32, kernel_size=9, strides=1, padding="same")(i)
    i = BatchNormalization()(i)
    i = relu()(i)
    i = SpatialDropout2D(0.1)(i)
    i = AveragePooling2D(pool_size=(1, 2))(i)

    # Channel Average Pooling and Reshaping
    def global_avg_pooling(x):
        return K.mean(x, axis=-1, keepdims=True)

    i = Lambda(global_avg_pooling)(i)

    # # Encoder block/Attention mechanisms
    i = MultiHeadAttention(num_heads=8, key_dim=100, dropout=0.3)(i, i)

    # # Flatten
    i = Flatten()(i)
    i = Dense(num_classes, activation="softmax")(i)

    return Model(inputs=input_layer, outputs=i)


# N blocks Transformer
def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    # Normalization and Attention
    x = LayerNormalization(epsilon=1e-6)(inputs)
    x = MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(
        x, x
    )
    x = Dropout(dropout)(x)
    res = x + inputs

    # Feed Forward Part
    x = LayerNormalization(epsilon=1e-6)(res)
    x = Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
    x = Dropout(dropout)(x)
    x = Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    return x + res


def build_model_Trasnformer_N_blocks(
    input_shape,
    head_size,
    num_heads,
    ff_dim,
    num_transformer_blocks,
    mlp_units,
    dropout=0,
    mlp_dropout=0,
):
    inputs = Input(shape=input_shape)
    x = inputs
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)
    x = GlobalAveragePooling1D(data_format="channels_first")(x)
    for dim in mlp_units:
        x = Dense(dim, activation="relu")(x)
        x = Dropout(mlp_dropout)(x)
    outputs = Dense(2)(x)
    return Model(inputs, outputs)


# == Residual CNN ==
def conv(i, filters=16, kernel_size=9, strides=1):
    i = Conv1D(
        filters=filters, kernel_size=kernel_size, strides=strides, padding="same"
    )(i)
    i = BatchNormalization()(i)
    i = leaky_relu()(i)
    i = SpatialDropout1D(0.1)(i)
    return i


def residual_unit(x, filters, layers=3):
    inp = x
    for i in range(layers):
        x = conv(x, filters)
    return Add([x, inp])


def conv_block(x, filters, strides):
    x = conv(x, filters)
    x = residual_unit(x, filters)
    if strides > 1:
        x = AveragePooling1D(strides, strides)(x)
    return x


def build_model(input_shape, num_classes):
    inp = Input(input_shape)
    x = inp
    x = conv_block(x, 16, 4)
    x = conv_block(x, 16, 4)
    x = conv_block(x, 32, 4)
    x = conv_block(x, 32, 4)
    x = Masking(mask_value=0)(x)
    x = GRU(128, recurrent_dropout=0.1)(x)
    x = Dense(num_classes, activation="softmax")(x)
    model = Model(inp, x)
    return model


# == CNN + GRU ==
def build_model_CNN_GRU(input_shape, num_classes):
    input_layer = Input(input_shape)
    i = Masking(mask_value=0)(input_layer)  # Masking
    # Conv1
    i = Conv1D(filters=16, kernel_size=9, strides=1, padding="same")(i)
    i = BatchNormalization()(i)
    i = relu()(i)
    i = SpatialDropout1D(0.1)(i)
    i = AveragePooling1D(4, 4)(i)
    # Conv2
    i = Conv1D(filters=32, kernel_size=9, strides=1, padding="same")(i)
    i = BatchNormalization()(i)
    i = relu()(i)
    i = SpatialDropout1D(0.1)(i)
    i = AveragePooling1D(4, 4)(i)
    # Conv3
    i = Conv1D(filters=64, kernel_size=9, strides=1, padding="same")(i)
    i = BatchNormalization()(i)
    i = relu()(i)
    i = SpatialDropout1D(0.1)(i)
    i = AveragePooling1D(4, 4)(i)

    # i = keras.layers.GRU(128, recurrent_dropout=0.1)(i)
    # i = keras.layers.Dense(num_classes, activation='softmax')(i)

    return Model(inputs=input_layer, outputs=i)


# == Transformer fixed split ==
def get_positional_encoding(seq_length, d_model):
    position = np.arange(seq_length)[:, np.newaxis]
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
    pe = np.zeros((seq_length, d_model))
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)
    pe = pe[np.newaxis, ...]
    return tf.cast(pe, dtype=tf.float32)


def build_model(
    input_shape, num_classes, num_heads=9, num_encoders=3, dff=512, dropout_rate=0.3
):
    seq_length, d_model = input_shape[1], input_shape[2]

    # Input layer
    inputs = Input(shape=input_shape[1:])

    # Positional encoding
    positional_encoding = get_positional_encoding(seq_length, d_model)
    x = Add()([inputs, positional_encoding])

    # Encoder layers
    for _ in range(num_encoders):
        # Multi-head-attention
        attn_output = MultiHeadAttention(num_heads=num_heads, key_dim=d_model)(x, x)
        # Residual connection
        x = Add()([x, attn_output])
        # Dropout
        x = Dropout(dropout_rate)(x)
        # Layer normalization
        x = LayerNormalization(epsilon=1e-6)(x)

        # Feed forward
        ffn_output = Dense(dff, activation="relu")(x)
        ffn_output = Dense(d_model)(ffn_output)
        # Residual connection
        x = Add()([x, ffn_output])
        # Dropout
        x = Dropout(dropout_rate)(x)
        # Layer normalization
        x = LayerNormalization(epsilon=1e-6)(x)

    # Global average pooling
    x = GlobalAveragePooling1D()(x)

    # Final output layer
    outputs = Dense(num_classes, activation="softmax")(x)

    return Model(inputs=inputs, outputs=outputs)
