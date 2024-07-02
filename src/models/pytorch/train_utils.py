import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

num_epochs = 100
batch_size = 32
learning_rate = 0.001

def train_multiclassification_model(X, Y, model):
    # Split the balanced dataset into training and validation sets
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

    # Convert to torch tensors with the appropriate types
    input_data_train = torch.tensor(X_train, dtype=torch.float32)
    target_data_train = torch.tensor(Y_train, dtype=torch.long)  # Ensure Y_train is of type torch.long
    input_data_val = torch.tensor(X_val, dtype=torch.float32)
    target_data_val = torch.tensor(Y_val, dtype=torch.long)  # Ensure Y_val is of type torch.long

    # Create PyTorch datasets and loaders
    train_dataset = TensorDataset(input_data_train, target_data_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
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