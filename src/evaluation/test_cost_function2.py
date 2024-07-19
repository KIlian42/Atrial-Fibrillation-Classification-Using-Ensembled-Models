import os
import pandas as pd
import tensorflow as tf

# Load the class weights from the uploaded file
weights_path = os.path.join(path, "prepared/Physionet2021_evaluation/weights.csv")
class_weights_df = pd.read_csv(weights_path)

# Reorder rows according to new_order
new_order = [1, 2, 5, 6, 7, 8, 9, 10, 11, 12, 17, 13, 18, 14, 0, 3, 15, 16, 4, 19, 20, 21, 22, 23, 24, 25]
class_weights_df = class_weights_df.iloc[new_order].reset_index(drop=True)
# Set the index and convert to numpy array
class_weights = class_weights_df.set_index(class_weights_df.columns[0]).values
# Calculate the inverse of class weights
inverse_class_weights = 1 / class_weights
# print(inverse_class_weights)

def binary_cross_entropy(y_true, y_pred):
    return tf.reduce_mean(tf.keras.losses.binary_crossentropy(y_true, y_pred))

def sparsity_loss(y_pred):
    return -4 * y_pred * (y_pred - 1)

def challenge_loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    # Compute confusion matrix entries
    a_ij = tf.matmul(tf.expand_dims(y_pred, -1), tf.expand_dims(y_pred, 1))

    # Compute weighted sum
    cl_loss = tf.reduce_sum(class_weights * a_ij) # 1 = inverse_class_weights
    return cl_loss

# Custom loss function
def custom_loss(y_true, y_pred):
    bce = binary_cross_entropy(y_true, y_pred)
    cl = challenge_loss(y_true, y_pred)
    sl = sparsity_loss(y_pred)
    return bce - cl # + sl
    # return bce - cl + sl

y_true = [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
y_pred = [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
y_true = tf.constant(y_true)
y_pred = tf.constant(y_pred)
print(custom_loss(y_true, y_pred))