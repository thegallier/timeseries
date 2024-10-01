import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import torch.nn.functional as F
from sklearn.metrics import cohen_kappa_score
from torch.utils.tensorboard import SummaryWriter
from itertools import product
import time

# Parameters
num_timesteps = 5000  # Reduced for practical purposes
num_securities = 10
num_features_per_security = 4
num_classes = 3
num_features = num_securities * num_features_per_security

# Generate timestamps
timestamps = np.arange(num_timesteps)

# Generate random data for the primary dataset
X_data = np.random.rand(num_timesteps, num_features).astype(np.float32)
y_data = np.random.randint(0, num_classes, size=(num_timesteps, num_securities)).astype(np.int64)

# Generate the second dataset
# 3 strings and 2 floats per timestamp
str_columns = ['str1', 'str2', 'str3']
float_columns = ['float1', 'float2']
second_dataset = {
    'timestamp': timestamps,
    'str1': np.random.choice(['A', 'B', 'C'], num_timesteps),
    'str2': np.random.choice(['D', 'E', 'F'], num_timesteps),
    'str3': np.random.choice(['G', 'H', 'I'], num_timesteps),
    'float1': np.random.rand(num_timesteps),
    'float2': np.random.rand(num_timesteps),
}

# Encode string columns
label_encoders = {}
for col in str_columns:
    le = LabelEncoder()
    second_dataset[col] = le.fit_transform(second_dataset[col])
    label_encoders[col] = le

# Combine all features from the second dataset
second_X_data = np.column_stack([second_dataset[col] for col in str_columns + float_columns]).astype(np.float32)

# Min-max scaling for both datasets
scaler_X = MinMaxScaler()
X_data = scaler_X.fit_transform(X_data)

scaler_second_X = MinMaxScaler()
second_X_data = scaler_second_X.fit_transform(second_X_data)

def add_positional_encoding(X, timestamps, option='shared'):
    """
    Add positional encoding to the feature matrix.

    Parameters:
    - X: Feature matrix.
    - timestamps: Array of timestamps.
    - option: 'shared' or 'per_security'.

    Returns:
    - X with positional encoding added.
    """
    if option == 'shared':
        pe = np.sin(timestamps[:, None] / 10000 ** (np.arange(X.shape[1]) / X.shape[1]))
        X_pe = X + pe.astype(np.float32)
    elif option == 'per_security':
        pe_list = []
        for i in range(num_securities):
            pe = np.sin(timestamps[:, None] / 10000 ** (np.arange(num_features_per_security) / num_features_per_security))
            pe_list.append(pe)
        pe_concat = np.hstack(pe_list)
        X_pe = X + pe_concat.astype(np.float32)
    else:
        X_pe = X  # No positional encoding
    return X_pe

# Apply positional encoding
positional_encoding_option = 'shared'  # 'shared' or 'per_security'
X_data = add_positional_encoding(X_data, timestamps, positional_encoding_option)

def create_windows(X1, X2, y, window_size, horizon):
    """
    Create sliding windows for time series data.

    Parameters:
    - X1: Primary dataset features.
    - X2: Secondary dataset features.
    - y: Target variable.
    - window_size: Size of the window.
    - horizon: Prediction horizon.

    Returns:
    - Tuples of windows for X1, X2, and y.
    """
    X1_windows = []
    X2_windows = []
    y_windows = []
    for i in range(len(X1) - window_size - horizon + 1):
        X1_windows.append(X1[i:i+window_size])
        X2_windows.append(X2[i:i+window_size])
        y_windows.append(y[i+window_size+horizon-1])
    return np.array(X1_windows), np.array(X2_windows), np.array(y_windows)

# Define window sizes
window_size = 20
horizon = 1

# Create windows
X1_windows, X2_windows, y_windows = create_windows(X_data, second_X_data, y_data, window_size, horizon)

# Split into train and test sets
train_ratio = 0.8
train_size = int(len(X1_windows) * train_ratio)

X1_train = X1_windows[:train_size]
X2_train = X2_windows[:train_size]
y_train = y_windows[:train_size]

X1_test = X1_windows[train_size:]
X2_test = X2_windows[train_size:]
y_test = y_windows[train_size:]

# Convert to PyTorch tensors
X1_train_tensor = torch.tensor(X1_train)
X2_train_tensor = torch.tensor(X2_train)
y_train_tensor = torch.tensor(y_train)

X1_test_tensor = torch.tensor(X1_test)
X2_test_tensor = torch.tensor(X2_test)
y_test_tensor = torch.tensor(y_test)

class TimeSeriesDataset(Dataset):
    """
    Custom Dataset for time series data with two feature sets.

    Parameters:
    - X1: Primary dataset features.
    - X2: Secondary dataset features.
    - y: Target variable.
    """
    def __init__(self, X1, X2, y):
        self.X1 = X1.float()
        self.X2 = X2.float()
        self.y = y.long()
    def __len__(self):
        return len(self.X1)
    def __getitem__(self, idx):
        return self.X1[idx], self.X2[idx], self.y[idx]

# DataLoaders
batch_size = 64

train_dataset = TimeSeriesDataset(X1_train_tensor, X2_train_tensor, y_train_tensor)
test_dataset = TimeSeriesDataset(X1_test_tensor, X2_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

class MatrixRegressionModel(nn.Module):
    """
    Matrix Regression Model: Y = A * X * B

    Advantages:
    - Simple and interpretable.
    - Fast training.

    Disadvantages:
    - Limited in capturing complex patterns.
    """
    def __init__(self, num_securities, num_features):
        super(MatrixRegressionModel, self).__init__()
        self.A = nn.Parameter(torch.randn(1, num_securities))
        self.B = nn.Parameter(torch.randn(num_features, num_classes))
    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten over time
        out = self.A @ x @ self.B  # Shape: (batch_size, num_classes)
        out = out.view(-1, num_securities, num_classes)
        return out

class LogisticRegressionModel(nn.Module):
    """
    Logistic Regression Model.

    Advantages:
    - Simple and interpretable.
    - Good baseline model.

    Disadvantages:
    - Assumes linear relationship.
    - May underfit complex data.
    """
    def __init__(self, input_size, num_securities, num_classes):
        super(LogisticRegressionModel, self).__init__()
        self.num_securities = num_securities
        self.num_classes = num_classes
        self.linear = nn.Linear(input_size, num_securities * num_classes)
    def forward(self, x):
        x = x.view(x.size(0), -1)
        out = self.linear(x)
        out = out.view(-1, self.num_securities, self.num_classes)
        return out

class CNNModel(nn.Module):
    """
    Convolutional Neural Network Model.

    Advantages:
    - Captures local temporal patterns.
    - Efficient computation.

    Disadvantages:
    - Limited in capturing long-term dependencies.
    """
    def __init__(self, num_features, num_securities, num_classes):
        super(CNNModel, self).__init__()
        self.num_securities = num_securities
        self.num_classes = num_classes
        self.conv1 = nn.Conv1d(in_channels=num_features, out_channels=64, kernel_size=3)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(128, num_securities * num_classes)
    def forward(self, x):
        x = x.permute(0, 2, 1)  # (batch_size, num_features, window_size)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = torch.mean(x, dim=2)  # Global average pooling
        x = self.fc(x)
        out = x.view(-1, self.num_securities, self.num_classes)
        return out

class LSTMModel(nn.Module):
    """
    Long Short-Term Memory Model.

    Advantages:
    - Captures long-term dependencies.
    - Suitable for sequential data.

    Disadvantages:
    - Computationally intensive.
    - Prone to overfitting.
    """
    def __init__(self, input_size, hidden_size, num_layers, num_securities, num_classes):
        super(LSTMModel, self).__init__()
        self.num_securities = num_securities
        self.num_classes = num_classes
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_securities * num_classes)
    def forward(self, x):
        h0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        c0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]  # Last time step
        out = self.fc(out)
        out = out.view(-1, self.num_securities, self.num_classes)
        return out

class TransformerModel(nn.Module):
    """
    Transformer Model.

    Advantages:
    - Captures global dependencies.
    - Parallel computation.

    Disadvantages:
    - Requires large datasets.
    - Computationally intensive.
    """
    def __init__(self, num_features, num_securities, num_classes):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Linear(num_features, 128)
        self.pos_encoder = PositionalEncoding(128)
        encoder_layers = nn.TransformerEncoderLayer(d_model=128, nhead=8)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=2)
        self.decoder = nn.Linear(128, num_securities * num_classes)
    def forward(self, x):
        x = x.view(x.size(0), x.size(1), -1)  # (batch_size, window_size, num_features)
        x = self.embedding(x)
        x = self.pos_encoder(x)
        x = x.permute(1, 0, 2)  # (sequence_length, batch_size, embedding_dim)
        output = self.transformer_encoder(x)
        output = output[-1, :, :]  # Last output
        output = self.decoder(output)
        output = output.view(-1, self.num_securities, num_classes)
        return output

class PositionalEncoding(nn.Module):
    """
    Positional Encoding for Transformer.

    Adds positional information to the embeddings.
    """
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.sin(position * div_term)
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)
        self.register_buffer('pe', pe)
    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return x

class SimpleRNNModel(nn.Module):
    """
    Simple Recurrent Neural Network Model.

    Advantages:
    - Captures sequential dependencies.
    - Simpler than LSTM.

    Disadvantages:
    - Struggles with long-term dependencies.
    - May suffer from vanishing gradients.
    """
    def __init__(self, input_size, hidden_size, num_securities, num_classes):
        super(SimpleRNNModel, self).__init__()
        self.num_securities = num_securities
        self.num_classes = num_classes
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_securities * num_classes)
    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), self.rnn.hidden_size).to(x.device)
        out, _ = self.rnn(x, h0)
        out = out[:, -1, :]  # Last time step
        out = self.fc(out)
        out = out.view(-1, self.num_securities, self.num_classes)
        return out

class CombinedModel(nn.Module):
    """
    Combined Model: Merges outputs from two models.

    Advantages:
    - Leverages multiple data sources.
    - Potentially better performance.

    Disadvantages:
    - More complex.
    - Computationally intensive.
    """
    def __init__(self, model1, model2, num_classes, num_securities):
        super(CombinedModel, self).__init__()
        self.model1 = model1
        self.model2 = model2
        self.fc = nn.Linear(num_classes * 2, num_classes)
        self.num_securities = num_securities
    def forward(self, x1, x2):
        outputs1 = self.model1(x1)
        outputs2 = self.model2(x2)
        outputs = torch.cat((outputs1, outputs2), dim=2)
        batch_size = outputs.size(0)
        outputs = outputs.view(-1, outputs.size(2))
        final_output = self.fc(outputs)
        final_output = final_output.view(batch_size, self.num_securities, -1)
        return final_output

def train_model(model, train_loader, criterion, optimizer, num_epochs, device, writer):
    """
    Training loop for the model.

    Parameters:
    - model: The model to train.
    - train_loader: DataLoader for training data.
    - criterion: Loss function.
    - optimizer: Optimizer.
    - num_epochs: Number of epochs.
    - device: Computation device.
    - writer: TensorBoard SummaryWriter.
    """
    model = model.to(device)
    global_step = 0
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for X1_batch, X2_batch, y_batch in train_loader:
            X1_batch = X1_batch.to(device)
            X2_batch = X2_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X1_batch, X2_batch)
            loss = 0
            for i in range(num_securities):
                loss += criterion(outputs[:, i, :], y_batch[:, i])
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            writer.add_scalar('Training Loss', loss.item(), global_step)
            global_step += 1
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')
        # Save checkpoint
        torch.save(model.state_dict(), f'checkpoint_epoch_{epoch+1}.pth')

def evaluate_model(model, test_loader, device):
    """
    Evaluate the model on the test set.

    Parameters:
    - model: The model to evaluate.
    - test_loader: DataLoader for test data.
    - device: Computation device.

    Returns:
    - Cohen's kappa score.
    """
    model = model.to(device)
    model.eval()
    total = 0
    correct = 0
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for X1_batch, X2_batch, y_batch in test_loader:
            X1_batch = X1_batch.to(device)
            X2_batch = X2_batch.to(device)
            y_batch = y_batch.to(device)
            outputs = model(X1_batch, X2_batch)
            _, predicted = torch.max(outputs.data, 2)
            total += y_batch.numel()
            correct += (predicted == y_batch).sum().item()
            all_preds.extend(predicted.cpu().numpy().flatten())
            all_targets.extend(y_batch.cpu().numpy().flatten())
    accuracy = 100 * correct / total
    kappa = cohen_kappa_score(all_targets, all_preds)
    print(f'Accuracy on test set: {accuracy:.2f}%')
    print(f"Cohen's Kappa: {kappa:.4f}")
    return kappa

def hyperparameter_tuning(hyperparams, train_loader, test_loader, device):
    """
    Perform hyperparameter tuning.

    Parameters:
    - hyperparams: Dictionary of hyperparameters to try.
    - train_loader: DataLoader for training data.
    - test_loader: DataLoader for test data.
    - device: Computation device.
    """
    best_kappa = -1
    best_params = None
    for params in product(*hyperparams.values()):
        param_dict = dict(zip(hyperparams.keys(), params))
        print(f"Trying hyperparameters: {param_dict}")
        model1 = TransformerModel(num_features=num_features, num_securities=num_securities, num_classes=num_classes)
        model2 = LSTMModel(input_size=second_X_data.shape[1], hidden_size=param_dict['hidden_size'],
                           num_layers=param_dict['num_layers'], num_securities=num_securities, num_classes=num_classes)
        combined_model = CombinedModel(model1, model2, num_classes, num_securities)
        optimizer = torch.optim.Adam(combined_model.parameters(), lr=param_dict['learning_rate'])
        criterion = nn.CrossEntropyLoss()
        writer = SummaryWriter()
        train_model(combined_model, train_loader, criterion, optimizer, param_dict['num_epochs'], device, writer)
        kappa = evaluate_model(combined_model, test_loader, device)
        writer.close()
        if kappa > best_kappa:
            best_kappa = kappa
            best_params = param_dict
    print(f"Best Cohen's Kappa: {best_kappa:.4f} with parameters: {best_params}")

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters for tuning
hyperparams = {
    'hidden_size': [64, 128],
    'num_layers': [1, 2],
    'learning_rate': [0.001, 0.0001],
    'num_epochs': [5]
}

# Start hyperparameter tuning
hyperparameter_tuning(hyperparams, train_loader, test_loader, device)

model = LSTMModel(input_size=num_features, hidden_size=128, num_layers=2, num_securities=num_securities, num_classes=num_classes)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()
writer = SummaryWriter()
train_model(model, train_loader, criterion, optimizer, num_epochs=5, device=device, writer=writer)

evaluate_model(model, test_loader, device)
writer.close()
