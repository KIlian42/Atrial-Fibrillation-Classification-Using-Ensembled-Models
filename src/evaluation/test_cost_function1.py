import tensorflow as tf
from tensorflow.keras.losses import Loss
import numpy as np

class WeightedCategoricalCrossentropy(Loss):
    def __init__(self, class_weight_matrix):
        super(WeightedCategoricalCrossentropy, self).__init__()
        self.class_weight_matrix = tf.constant(class_weight_matrix, dtype=tf.float32)

    def call(self, y_true, y_pred):
        # Convert y_true to one-hot encoding
        # y_true = tf.cast(tf.squeeze(y_true), tf.int32)
        # print("\ny_true:\n", y_true.numpy())
        y_true_one_hot = tf.one_hot(y_true, depth=y_pred.shape[-1])
        print("\ny_true_one_hot:\n", y_true_one_hot.numpy())

        # Clip predictions to prevent log(0)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        print("\ny_pred:\n", y_pred.numpy())

        # Compute the categorical cross-entropy loss
        loss = tf.keras.losses.categorical_crossentropy(y_true_one_hot, y_pred)
        print("\nloss:\n", loss.numpy())

        # Gather the weights for the true class labels
        weights = tf.gather(self.class_weight_matrix, y_true)
        print("\ngathered weights:\n", weights.numpy())

        # ==============================
        y_pred = tf.cast(y_pred, tf.float32)
        # Compute the weights to apply to the loss
        multiplied = weights * y_pred
        print("\nmultiplied:\n", multiplied.numpy())
        weights = tf.reduce_sum(weights * y_pred, axis=-1)
        print("\nweights:\n", weights.numpy())

        # Cast values
        weights = tf.cast(weights, tf.double)
        y_true_one_hot = tf.cast(y_true_one_hot, tf.float32)

        # Apply the weights to the loss
        weighted_loss = loss * weights
        print(weighted_loss.numpy())

        return tf.reduce_mean(weighted_loss)
        # return tf.reduce_sum(weighted_losses)


# Example class weight matrix for 5 classes with emphasis on misclassifications
class_weight_matrix = [
    [0.0, 1.0, 1.0],
    [1.0, 0.0, 1000.0],
    [1.0, 1000.0, 0.0],
]

loss_function = WeightedCategoricalCrossentropy(class_weight_matrix)

# print("\nOutput:\n", loss_function.call(np.array([0, 1, 1, 1, 1]), np.array([[0.99, 0.005, 0.005], [0.005, 0.005, 0.99], [0.005, 0.005, 0.99], [0.005, 0.005, 0.99], [0.005, 0.005, 0.99]])).numpy())
# print("\nOutput:\n", loss_function.call(np.array([0, 1]), np.array([[0.99, 0.005, 0.005], [0.005, 0.005, 0.99]])).numpy())
# print("\nOutput:\n", loss_function.call(np.array([0]), np.array([[0.99, 0.005, 0.005]])).numpy())
print("\nOutput:\n", loss_function.call(np.array([[0, 1, 0], [1, 0, 0]]), np.array([[0.005, 0.005, 0.99], [0.005, 0.005, 0.99]])).numpy())