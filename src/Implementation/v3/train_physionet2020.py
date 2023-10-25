# https://github.com/hadaev8/physionet_2017_rcrnn/blob/master/tf_keras_RCNN_physionet_2017_cross_val.ipynbç
# https://github.com/physionetchallenges/physionetchallenges.github.io/blob/master/2020/Dx_map.csv
# https://gist.github.com/antonior92/da7f1e884428aa35fda20e2820181280

import h5py
import numpy as np

from tqdm import tqdm

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

import tensorflow as tf
import keras

import matplotlib.pyplot as plt

def pad_or_truncate_lists(lst, n):
    return lst[:n] + [0] * (n - len(lst))
    
def conv(i, filters=16, kernel_size=9, strides=1):
    i = keras.layers.Conv1D(filters=filters, kernel_size=kernel_size, strides=strides, padding='same')(i)
    i = keras.layers.BatchNormalization()(i)
    i = keras.layers.LeakyReLU()(i)
    i = keras.layers.SpatialDropout1D(0.1)(i)
    return i

def residual_unit(x, filters, layers=3):
    inp = x
    for i in range(layers):
        x = conv(x, filters)
    return keras.layers.add([x, inp])

def conv_block(x, filters, strides):
    x = conv(x, filters)
    x = residual_unit(x, filters)
    if strides > 1:
        x = keras.layers.AveragePooling1D(strides, strides)(x)
    return x

def get_model(num_classes):
    inp = keras.layers.Input(shape=(X.shape[1], 1), dtype=tf.float32)

    x = inp
    x = conv_block(x, 16, 4)
    x = conv_block(x, 16, 4)
    x = conv_block(x, 32, 4)
    x = conv_block(x, 32, 4)
    x = keras.layers.Masking(mask_value=0)(x)
    x = keras.layers.GRU(128, recurrent_dropout=0.1)(x)
    x = keras.layers.Dense(num_classes, activation='softmax')(x)

    model = keras.models.Model(inp, x)
    return model

def main():
    global X
    global Y
    # Load data
    h5file =  h5py.File('physionet2020.h5', 'r')
    ID = list(h5file.keys())
    X = [list(h5file[key][1][:]) for key in tqdm(ID, desc="Load ECG data")]
    # Labels
    file = open("physionet2020-references.csv", "r")
    Y_labels = file.readlines()
    file.close()
    Y_labels = [label.replace("\n", "").split(",")[0] for label in tqdm(Y_labels, desc="Load ECG labels")]
    # Encode labels
    encoder = LabelEncoder()
    encoder.fit(Y_labels)
    Y = encoder.transform(Y_labels)
    print("Classes:", encoder.classes_) # => 4 classes in total

    print("Total training data before preprocssing:", len(ID))
    # Preprocess
    ID_preprocessed = []
    X_preprocessed = []
    Y_preprocessed = []
    Y_labels_preprocessed = []
    unique_length = []
    for _, data in enumerate(tqdm(zip(ID, X, Y, Y_labels), desc="Preprocess ECGs")):
        if data[3] == "164889003" or data[3] == "164890007" or data[3] == "426783006":
            current_label = data[2]
            current_label_text = data[3]
            unique_length.append(len(data[1]))
            ID_preprocessed.append(data[0])
            X_preprocessed.append(np.array(pad_or_truncate_lists(data[1], 5000)))
            Y_preprocessed.append(current_label)
            Y_labels_preprocessed.append(current_label_text)

    # print(set(unique_length))
    from collections import Counter
    count = Counter(unique_length)
    print(count)

    # Replace
    ID = ID_preprocessed
    X = X_preprocessed
    Y = Y_preprocessed
    Y_labels = Y_labels_preprocessed

    print("Total training data after preprocssing:", len(ID_preprocessed))
    
    # Convert to numpy
    X = np.array(X)
    Y = np.array(Y)

    # Split training & testing data
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42, shuffle=True)
    # Reshape data
    x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
    x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))
    # Shuffle/Permutate training data
    idx = np.random.permutation(len(x_train))
    x_train = x_train[idx]
    y_train = y_train[idx]

    # # =========================================================

    global batch_size
    batch_size = 128
    epochs = 2

    model = get_model(len(encoder.classes_))
    model.summary()

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            "best_model_physionet2020.h5", save_best_only=True, monitor="val_loss"
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=20, min_lr=0.005 # 0.005
        ),
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=50, verbose=1),
    ]
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["sparse_categorical_accuracy"],
    )
    history = model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        callbacks=callbacks,
        validation_split=0.2,
        verbose=1,
    )

    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(1, len(train_loss) + 1)
    plt.figure(figsize=(12, 6))
    plt.plot(epochs_range, train_loss, 'r', label='Training Loss')
    plt.plot(epochs_range, val_loss, 'b', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig("/Users/kiliankramer/Desktop/Training_loss_physionet2020.png", dpi=300, format='png', bbox_inches='tight')
    # plt.show()

    train_accuracy = history.history['sparse_categorical_accuracy']
    val_accuracy = history.history['val_sparse_categorical_accuracy']
    epochs_range = range(1, len(train_accuracy) + 1)
    plt.figure(figsize=(12, 6))
    plt.plot(epochs_range, train_accuracy, 'r', label='Training Accuracy')
    plt.plot(epochs_range, val_accuracy, 'b', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig("/Users/kiliankramer/Desktop/Training_accuracy_physionet2020.png", dpi=300, format='png', bbox_inches='tight')
    # plt.show()

if __name__ == '__main__':
    main()