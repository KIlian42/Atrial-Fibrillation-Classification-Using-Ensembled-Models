import warnings
warnings.filterwarnings("ignore")

import os
import multiprocessing
from tqdm import tqdm
import h5py
import pandas as pd

from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler

from collections import Counter
from sklearn.model_selection import train_test_split

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from util.preprocess_ecg import split_list_into_n_sublists, pad_or_truncate_ecg
from models.pytorch_transformer_multiclassification2 import Encoder

path = "/Users/kiliankramer/Desktop/Master Thesis/ImplementationNew/Atrial-Fibrillation-Classification-Using-Ensembled-Models/src/datasets/"
h5file = h5py.File(os.path.join(path, "physionet2017.h5"), "r")

def preprocess_ecgs(input_list, process, target_list, lock):
    for id in tqdm(input_list, desc=f"Preprocess ecgs ({process+1}. process)", position=process):
        # with lock:
        target_list[id] = pad_or_truncate_ecg(list(h5file[id]), 9000)

def main():
    manager = multiprocessing.Manager()
    X_dict = manager.dict()
    lock = manager.Lock()
    num_processes = multiprocessing.cpu_count()
    print(f"CPUs available: {num_processes}")
    # IDs = split_list_into_n_sublists(list(h5file.keys()), num_processes)
    IDs = split_list_into_n_sublists(list(h5file.keys()), num_processes)
    processes = []
    for process in range(num_processes):
        p = multiprocessing.Process(target=preprocess_ecgs, args=(IDs[process], process, X_dict, lock))
        processes.append(p)
        p.start()
    for process in processes:
        process.join()
    X_dict = dict(X_dict)

    Y_dict = {}
    label_dict = {"N": 0, "A": 1, "O": 2, "~": 3}
    labels_df = pd.read_csv(os.path.join(path, "physionet2017_references.csv"), sep=",")
    pbar = tqdm(total=len(labels_df), desc="Load ECG labels", position=0, leave=True)
    for _, row in labels_df.iterrows():
        Y_dict[row["id"]] = label_dict[row["label"]]
        pbar.update(1)

    Y_values = Y_dict.values()
    counts = Counter(Y_values)
    print(f"Label distribution: {counts}")

    X = []
    Y = []
    for id in X_dict:
        X.append(np.array(X_dict[id]))
        Y.append(Y_dict[id])
    X = np.array(X)
    Y = np.array(Y)

    count_train = Counter(Y)
    CLASS_BALANCE = 2000
    undersampling_strategy = {}
    oversampling_strategy = {}
    for i in count_train:
        if count_train[i] > CLASS_BALANCE:
            undersampling_strategy[i] = CLASS_BALANCE
        elif count_train[i] <= CLASS_BALANCE:
            oversampling_strategy[i] = CLASS_BALANCE
    print(f"Trainset undersampling_strategy: {undersampling_strategy}")
    print(f"Trainset oversampling_strategy: {oversampling_strategy}")
    under = RandomUnderSampler(sampling_strategy=undersampling_strategy)
    over = RandomOverSampler(sampling_strategy=oversampling_strategy)
    steps = [("u", under), ("o", over)]
    pipeline = Pipeline(steps=steps)
    X, Y = pipeline.fit_resample(X, Y)
    X = np.array(X)
    Y = np.array(Y)
    count_train = Counter(Y)
    print(f"Label distribution: {count_train}")

    X = X.reshape(-1, 30, 300)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    num_classes = 4
    num_layers = 1
    max_sequence_length = 30
    d_model = 300
    num_heads = 10
    drop_prob = 0.1
    ffn_hidden = 24

    num_epochs = 100
    batch_size = 32
    learning_rate = 0.001

    model = Encoder(num_classes, max_sequence_length, d_model, ffn_hidden, num_heads, drop_prob, num_layers)

    def count_trainable_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {count_trainable_parameters(model)}")

    # input_data = torch.randn(1000, max_sequence_length, d_model)
    # target_data = torch.randint(0, 2, (1000, 1))

    # == Train model ==

    print(f"Shapes: {X.shape}, {Y.shape}")

    # Split the balanced dataset into training and validation sets
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

    # Create PyTorch datasets and loaders
    input_data_train = torch.tensor(X_train, dtype=torch.float32)
    target_data_train = torch.tensor(Y_train, dtype=torch.long)
    train_dataset = TensorDataset(input_data_train, target_data_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    input_data_val = torch.tensor(X_val, dtype=torch.float32)
    target_data_val = torch.tensor(Y_val, dtype=torch.long)
    val_dataset = TensorDataset(input_data_val, target_data_val)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # To store accuracy and loss values
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    # Early stopping parameters
    patience = 100
    best_val_loss = float('inf')
    early_stop_counter = 0

    # Move the model to the device
    model.to(device)

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            correct_train += (predicted == targets).sum().item()
            total_train += targets.size(0)

        train_loss /= len(train_loader.dataset)
        train_accuracy = correct_train / total_train

        # Validation phase
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                correct_val += (predicted == targets).sum().item()
                total_val += targets.size(0)

        val_loss /= len(val_loader.dataset)
        val_accuracy = correct_val / total_val

        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Train Acc: {train_accuracy:.4f}, Val Acc: {val_accuracy:.4f}')

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_counter = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print("Early stopping")
                break

        # Step the scheduler
        scheduler.step()

    print('Finished Training')


if __name__ == "__main__":
    main()