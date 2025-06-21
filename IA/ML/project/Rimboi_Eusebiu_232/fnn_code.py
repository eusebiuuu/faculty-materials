
import cv2
import os
import numpy as np
import pandas as pd
import torch

image_idx_train = {}
idx_train = 0
image_idx_validation = {}
idx_validation = 0
image_idx_test = {}
idx_test = 0

PATH = "./fmi-ml2025-competition-dataset/"

def load_and_resize_images(folder_path, image_idx, idx, target_size=(80, 80)):
    images = []
    for filename in os.listdir(folder_path):

        image_idx[filename[0:-4]] = idx
        idx += 1

        if filename.endswith('.png'):
            # Read image (OpenCV reads as BGR by default)
            img = cv2.imread(os.path.join(folder_path, filename))

            # Convert to RGB if needed (depends on your model)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Resize image
            img = cv2.resize(img, target_size)

            img = np.transpose(img, (2, 0, 1))

            images.append(img)

    return np.array(images)


train_folder = f"{PATH}train"
validation_folder = f'{PATH}validation'
test_folder = f'{PATH}test'

train_images = load_and_resize_images(train_folder, image_idx_train, idx_train)
validation_images = load_and_resize_images(validation_folder, image_idx_validation, idx_validation)
test_images = load_and_resize_images(test_folder, image_idx_test, idx_test)

print(len(train_images), len(validation_images), len(test_images))


csv1_path = f'{PATH}train.csv'
csv2_path = f'{PATH}validation.csv'

train_csv = pd.read_csv(csv1_path)
validation_csv = pd.read_csv(csv2_path)

print("CSV1 columns:", train_csv.columns)
print("CSV2 columns:", validation_csv.columns)

train_labels = np.zeros((len(train_images)))
validation_labels = np.zeros((len(validation_images)))
train_labels = torch.tensor(train_labels, dtype=torch.long)
validation_labels = torch.tensor(validation_labels, dtype=torch.long)

for i in range(0, len(train_csv['image_id'])):
  train_labels[image_idx_train[train_csv['image_id'][i]]] = int(train_csv['label'][i])

for i in range(0, len(validation_csv['image_id'])):
  validation_labels[image_idx_validation[validation_csv['image_id'][i]]] = int(validation_csv['label'][i])

print(len(train_labels), len(validation_labels))

print(train_images[0].shape)

"""### Images normalization"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, Normalizer
from torch import tensor, float32, int8
import torch

import torch

def torch_channelwise_normalize(train_images_np, val_images_np, test_images_np, eps=1e-7):
    """
    Normalize training and validation images using channel-wise mean and std.

    Args:
        train_images_np (np.ndarray): shape (n, 3, H, W)
        val_images_np (np.ndarray): shape (n, 3, H, W)

    Returns:
        (Tensor, Tensor): normalized train and val images
    """
    # Convert to float32 tensors
    train_images = torch.tensor(train_images_np, dtype=torch.float32)
    val_images = torch.tensor(val_images_np, dtype=torch.float32)
    test_images = torch.tensor(test_images_np, dtype=torch.float32)

    # Compute per-channel mean and std over training set
    mean = train_images.mean(dim=(0, 2, 3), keepdim=True)
    std = train_images.std(dim=(0, 2, 3), keepdim=True) + eps  # prevent divide-by-zero

    # Apply normalization to both train and val
    train_normalized = (train_images - mean) / std
    val_normalized = (val_images - mean) / std # use train stats for validation
    test_normalized = (test_images - mean) / std

    print("Channel-wise mean:", mean.view(-1).tolist())
    print("Channel-wise std: ", std.view(-1).tolist())

    return train_normalized, val_normalized, test_normalized


standard_train, standard_validation, standard_test = torch_channelwise_normalize(train_images, validation_images, test_images)

"""### Data augmentation and loading"""

import torch
from torchvision import transforms


augmentation_transform = transforms.Compose([
    # transforms.ToPILImage(),  # Convert tensor to PIL Image
    transforms.RandomHorizontalFlip(p=0.5),  # 50% chance of flip
    transforms.RandomVerticalFlip(p=0.5),
    # transforms.RandomRotation(30),  # Rotate within ±30 degrees
    # transforms.RandomResizedCrop(244, scale=(0.7, 1.0)),  # Random zoom/crop
    # transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    # transforms.ToTensor(),
])


class AugmentedDataset(torch.utils.data.Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

train_dataset = AugmentedDataset(
    standard_train,
    train_labels,
    transform=augmentation_transform
)

validation_dataset = AugmentedDataset(
    standard_validation,
    validation_labels
)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
final_test_loader = torch.utils.data.DataLoader(standard_test, batch_size=64, shuffle=False)
validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=64, shuffle=False)

"""### Assure deterministic values"""

import random

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

"""### Feedforward Neural Network"""

import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()

        # First Hidden Layer
        self.first_layer = nn.Linear(3 * 80 * 80, 100) # FC1
        self.bn1 = nn.BatchNorm1d(100) # BatchNorm for 100 features
        self.dropout1 = nn.Dropout(p=0.2)

        # Second Hidden Layer
        self.second_layer = nn.Linear(100, 50) # FC2
        self.bn2 = nn.BatchNorm1d(50) # BatchNorm for 100 features
        self.dropout2 = nn.Dropout(p=0.2)

        # Third Hidden Layer (you had this commented out partially in your forward)
        self.third_layer = nn.Linear(50, 100) # FC3
        self.bn3 = nn.BatchNorm1d(100) # BatchNorm for 100 features
        self.dropout3 = nn.Dropout(p=0.2)

        # Output Layer
        self.output_layer = nn.Linear(100, 5) # FC4 (output for 5 classes)
        self._initialize_weights()

    def _initialize_weights(self):
        # Iterate over all modules in the network
        for m in self.modules():
            # Apply Xavier initialization to Linear layers
            if isinstance(m, nn.Linear):
                # You can choose either xavier_uniform_ or xavier_normal_
                # xavier_uniform_ draws from a uniform distribution
                init.xavier_uniform_(m.weight)
                # xavier_normal_ draws from a normal distribution
                # init.xavier_normal_(m.weight)

                # Initialize bias to zero (common practice)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            # You might also initialize BatchNorm layers, though PyTorch defaults are often good
            elif isinstance(m, nn.BatchNorm1d):
                init.constant_(m.weight, 1) # Gamma (scale) to 1
                init.constant_(m.bias, 0)   # Beta (shift) to 0

    def forward(self, x):
        x = self.flatten(x)

        # First Layer: Linear -> BatchNorm -> ReLU -> (Dropout optional)
        x = self.first_layer(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout1(x) # Dropout can be here or after ReLU based on preference/experimentation

        # Second Layer: Linear -> BatchNorm -> ReLU -> Dropout
        x = self.second_layer(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout2(x)

        # Third Layer: Linear -> BatchNorm -> ReLU -> (Dropout optional)
        x = self.third_layer(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.dropout3(x) # Dropout can be here or after ReLU based on preference/experimentation

        # Output Layer (no BatchNorm or ReLU here, typically)
        x = self.output_layer(x)
        return x

"""### Train Neural Network"""

import torch
from torch import nn

model_nn = NeuralNetwork()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_nn.to(device)
num_classes = 5

loss_fn = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model_nn.parameters(), lr=1e-2)
optimizer = torch.optim.Adam(model_nn.parameters(), lr=1e-3, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer, step_size=10, gamma=0.1  # reduce LR every 10 epochs
)

# --- Modified Training Function to return loss and accuracy ---
def train_loop(dataloader, model, loss_fn, optimizer):
    model.train()
    total_loss = 0
    correct_predictions = 0
    total_samples = 0

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Forward pass
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Accumulate metrics
        total_loss += loss.item() * len(X) # Multiply by batch size for total loss in batch
        correct_predictions += (pred.argmax(1) == y).type(torch.float).sum().item()
        total_samples += len(X)

        if batch % 200 == 0:
            current_loss = loss.item()
            print(f"Loss: {current_loss:>7f}  [{batch * len(X):>5d}/{len(dataloader.dataset):>5d}]")

    avg_epoch_loss = total_loss / total_samples
    epoch_accuracy = correct_predictions / total_samples
    return avg_epoch_loss, epoch_accuracy

# --- Modified Validation Function to return loss and accuracy ---
def validate_loop(dataloader, model, loss_fn):
    model.eval()
    total_loss = 0
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)

            total_loss += loss_fn(pred, y).item() * len(X)
            correct_predictions += (pred.argmax(1) == y).type(torch.float).sum().item()
            total_samples += len(X)

    avg_loss = total_loss / total_samples
    accuracy = correct_predictions / total_samples
    return avg_loss, accuracy


# --- Main Training Loop ---
num_epochs = 30

# Lists to store metrics for plotting
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

print("Starting training...")
for t in range(num_epochs):
    print(f"Epoch {t+1}\n-------------------------------")

    # Train
    epoch_train_loss, epoch_train_acc = train_loop(train_loader, model_nn, loss_fn, optimizer)
    train_losses.append(epoch_train_loss)
    train_accuracies.append(epoch_train_acc)

    # Validate
    epoch_val_loss, epoch_val_acc = validate_loop(validation_loader, model_nn, loss_fn)
    val_losses.append(epoch_val_loss)
    val_accuracies.append(epoch_val_acc)

    scheduler.step()

    print(f"Epoch {t+1} Summary: Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc*100:.2f}%, "
          f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc*100:.2f}%")
print("Done!")

# --- Plotting Loss and Accuracy History ---
import matplotlib.pyplot as plt
epochs_range = range(1, num_epochs + 1)

plt.figure(figsize=(14, 6)) # Increased figure size

# Plot Training and Validation Loss
plt.subplot(1, 2, 1) # 1 row, 2 columns, first plot
plt.plot(epochs_range, train_losses, 'o-', color='blue', label='Training Loss', markersize=4)
plt.plot(epochs_range, val_losses, 'o-', color='red', label='Validation Loss', markersize=4)
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# Plot Training and Validation Accuracy
plt.subplot(1, 2, 2) # 1 row, 2 columns, second plot
plt.plot(epochs_range, [acc * 100 for acc in train_accuracies], 'o-', color='blue', label='Training Accuracy', markersize=4)
plt.plot(epochs_range, [acc * 100 for acc in val_accuracies], 'o-', color='red', label='Validation Accuracy', markersize=4)
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.grid(True)

plt.tight_layout() # Adjust layout to prevent overlap
plt.show()


# --- Confusion Matrix Generation ---

from sklearn.metrics import confusion_matrix
import seaborn as sns

# Function to get true and predicted labels for confusion matrix
def get_predictions_and_true_labels(dataloader, model):
    model.eval()
    true_labels = []
    predicted_labels = []

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            predicted_class = pred.argmax(1) # Get the predicted class index

            true_labels.extend(y.cpu().numpy())
            predicted_labels.extend(predicted_class.cpu().numpy())

    return np.array(true_labels), np.array(predicted_labels)

print("\nGenerating Confusion Matrix for Validation Set...")
val_true_labels, val_predicted_labels = get_predictions_and_true_labels(validation_loader, model_nn)

# Generate Confusion Matrix
cm = confusion_matrix(val_true_labels, val_predicted_labels, labels=range(num_classes))

print("Confusion Matrix:")
print(cm)

# Visualize Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=range(num_classes), # Replace with actual class names if you have them (e.g., ['Real', 'Deepfake', ...])
            yticklabels=range(num_classes)) # Replace with actual class names if you have them

plt.title('Confusion Matrix (Validation Set)')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()
