import os
import pandas as pd
import tensorflow as tf

# Load the class weights from the uploaded file
weights_path = "/Users/kiliankramer/Desktop/Master Thesis/ImplementationNew/Atrial-Fibrillation-Classification-Using-Ensembled-Models/src/evaluation/weights.csv"
class_weights_df = pd.read_csv(weights_path)

# Reorder rows according to new_order
new_order_rows = [14, 0, 1, 15, 18, 2, 3, 4, 5, 6, 7, 8, 9, 11, 13, 16, 17, 10, 12, 19, 20, 21, 22, 23, 24, 25]
new_order_columns = ["426783006", "164889003", "164890007", "284470004|63593006", "427172004|17338001", "6374002", "426627000", "733534002|164909002", "713427006|59118001", "270492004", "713426002", "39732003", "445118002", "251146004", "698252002", "10370003", "365413008", "164947007", "111975006", "164917005", "47665007", "427393009", "426177001", "427084000", "164934002", "59931005"]  # Reorder by names
class_weights_df = class_weights_df.iloc[new_order_rows, :][new_order_columns].reset_index(drop=True)
class_weight_matrix = class_weights_df.values

def binary_cross_entropy(y_true, y_pred):
    return tf.reduce_mean(tf.keras.losses.binary_crossentropy(y_true, y_pred))

def sparsity_loss(y_pred):
    per_class_sparsity_penalty = -4 * y_pred * (y_pred - 1)
    return tf.reduce_sum(per_class_sparsity_penalty) # check for mean or sum

def challenge_loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    gathered_weights = []
    for index, y_t in enumerate(y_true):
        if y_t == 1:
            gathered_weights.append(tf.cast(tf.gather(class_weight_matrix, index), tf.float32))

    rewards = []
    for index, gathered_weight in enumerate(gathered_weights):
        rewards.append(gathered_weight * y_pred)
    
    print(rewards[0].numpy())
    cl_loss = tf.reduce_sum(rewards)
    return cl_loss

# Custom loss function
def custom_loss(y_true, y_pred):
    bce = binary_cross_entropy(y_true, y_pred)
    cl = challenge_loss(y_true, y_pred)
    sl = sparsity_loss(y_pred)
    return sl
    # return bce - cl + sl

y_true = [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
y_pred = [[1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
# y_pred = [[0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
y_true = tf.constant(y_true)
y_pred = tf.constant(y_pred)
print(custom_loss(y_true, y_pred))