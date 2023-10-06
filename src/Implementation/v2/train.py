import h5py
import numpy as np

from tqdm import tqdm

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from tensorflow import keras
from keras import layers

n_classes = 4

def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    # Normalization and Attention
    x = layers.LayerNormalization(epsilon=1e-6)(inputs)
    x = layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(x, x)
    x = layers.Dropout(dropout)(x)
    res = x + inputs

    # Feed Forward Part
    x = layers.LayerNormalization(epsilon=1e-6)(res)
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    return x + res

def build_model(
    input_shape,
    head_size,
    num_heads,
    ff_dim,
    num_transformer_blocks,
    mlp_units,
    dropout=0,
    mlp_dropout=0,
):
    inputs = keras.Input(shape=input_shape)
    x = inputs
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    x = layers.GlobalAveragePooling1D(data_format="channels_first")(x)
    for dim in mlp_units:
        x = layers.Dense(dim, activation="relu")(x)
        x = layers.Dropout(mlp_dropout)(x)
    outputs = layers.Dense(n_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs)
    
def main():
    # Load data
    h5file =  h5py.File('physionet2017.h5', 'r')
    ID = list(h5file.keys())
    X = [list(h5file[key][:]) for key in tqdm(ID, desc="Load ECG data")]
    # Labels
    file = open("physionet2017_references.csv", "r")
    Y_labels = file.readlines()
    file.close()
    Y_labels = [label.replace("\n", "").split(",")[1] for label in tqdm(Y_labels, desc="Load ECG labels")]
    # Encode labels
    encoder = LabelEncoder()
    encoder.fit(Y_labels)
    Y = encoder.transform(Y_labels)
    print("Classes:", encoder.classes_)
    # 4 classes in total

    # Preprocess / filter only 9000 sampled ECGs
    ID_preprocessed = []
    X_preprocessed = []
    Y_preprocessed = []
    Y_labels_preprocessed = []
    for index, data in enumerate(tqdm(zip(ID, X, Y, Y_labels), desc="Preprocess ECGs")):
        if len(data[1]) == 9000:
            ID_preprocessed.append(data[0])
            # X_preprocessed.append(divide_list(data[1], 30))
            X_preprocessed.append(data[1])
            Y_preprocessed.append(data[2])
            Y_labels_preprocessed.append(data[3])
    # Replace
    ID = ID_preprocessed
    X = X_preprocessed
    Y = Y_preprocessed
    Y_labels = Y_labels_preprocessed

    # Convert data into PyTorch tensors
    # X = torch.tensor(X, dtype=torch.float32)
    # Y = torch.tensor(Y, dtype=torch.int32)
    # Y = Y.type(torch.LongTensor)

    # # Split training and testing data
    # X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42, shuffle=True)
    # print("Training size:", X_train.shape[0], "// Training shape:", X_train.shape)
    # print("Testing size:", X_test.shape[0], "// Testing shape:", X_test.shape)

    # # Create DataLoader
    # train_loader = DataLoader(list(zip(X_train, Y_train)), shuffle=True, batch_size=64)
    # test_loader = DataLoader(list(zip(X_test, Y_test)), shuffle=True, batch_size=64)

    # # Defining model and training options
    # global device
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print("Using device:", device, f"({torch.cuda.get_device_name(device)})" if torch.cuda.is_available() else "")

    # # =========================================================

    # Convert to numpy
    X = np.array([np.array(inner_list) for inner_list in X])
    Y = np.array([np.array(inner_list) for inner_list in Y])

    # Split training & testing data
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42, shuffle=True)

    # Reshape data
    x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
    x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

    # Shuffle/Permutate training data
    idx = np.random.permutation(len(x_train))
    x_train = x_train[idx]
    y_train = y_train[idx]

    # Build the model
    input_shape = x_train.shape[1:]

    model = build_model(
        input_shape,
        head_size=256,
        num_heads=2,
        ff_dim=4,
        num_transformer_blocks=1,
        mlp_units=[128],
        mlp_dropout=0.4,
        dropout=0.25,
    )

    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        metrics=["sparse_categorical_accuracy"],
    )
    model.summary()

    callbacks = [keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)]

    model.fit(
        x_train,
        y_train,
        validation_split=0.2,
        epochs=200,
        batch_size=5,
        callbacks=callbacks,
    )

    model.evaluate(x_test, y_test, verbose=1)


if __name__ == '__main__':
    main()