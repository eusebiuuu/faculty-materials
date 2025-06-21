
import os
import random

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init # For weight initialization
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

DATASET_PATH = "./fmi-ml2025-competition-dataset/"
IMAGE_TARGET_SIZE = (80, 80)
NUM_CLASSES = 5

BATCH_SIZE = 32
NUM_EPOCHS = 30
SWITCH_OPTIMIZER_EPOCH = NUM_EPOCHS
LEARNING_RATE_ADAM = 1e-3
WEIGHT_DECAY_ADAM = 1e-5
LEARNING_RATE_SGD = 0.01
MOMENTUM_SGD = 0.9
WEIGHT_DECAY_SGD = 1e-3


SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def load_and_resize_images(folder_path: str, target_size: tuple) -> tuple[np.ndarray, dict]:
    images = []
    image_id_to_idx = {}
    current_idx = 0

    for filename in os.listdir(folder_path):
        if filename.endswith('.png'):
            image_id = filename[0:-4] # Remove .png extension
            image_id_to_idx[image_id] = current_idx
            current_idx += 1

            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path) # OpenCV reads as BGR by default

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Convert to RGB
            img = cv2.resize(img, target_size) # Resize image to (width, height)

            # Transpose to (C, H, W) for PyTorch compatibility
            img = np.transpose(img, (2, 0, 1))
            images.append(img)

    return np.array(images), image_id_to_idx

def load_labels_from_csv(csv_path: str, image_id_to_idx: dict, num_images: int) -> torch.Tensor:

    labels_df = pd.read_csv(csv_path)
    labels_tensor = torch.zeros(num_images, dtype=torch.long)

    for _, row in labels_df.iterrows():
        image_id = row['image_id']
        label = int(row['label'])
        if image_id in image_id_to_idx:
            labels_tensor[image_id_to_idx[image_id]] = label
    return labels_tensor

# Load and preprocess images
train_images_np, train_image_ids_map = load_and_resize_images(f"{DATASET_PATH}train", IMAGE_TARGET_SIZE)
validation_images_np, val_image_ids_map = load_and_resize_images(f'{DATASET_PATH}validation', IMAGE_TARGET_SIZE)
test_images_np, test_image_ids_map = load_and_resize_images(f'{DATASET_PATH}test', IMAGE_TARGET_SIZE)

print(f"Loaded {len(train_images_np)} training images, "
      f"{len(validation_images_np)} validation images, "
      f"and {len(test_images_np)} test images.")
print(f"Shape of first training image: {train_images_np[0].shape}")


# Load and map labels
train_labels = load_labels_from_csv(f'{DATASET_PATH}train.csv', train_image_ids_map, len(train_images_np))
validation_labels = load_labels_from_csv(f'{DATASET_PATH}validation.csv', val_image_ids_map, len(validation_images_np))

print(f"Loaded {len(train_labels)} training labels and {len(validation_labels)} validation labels.")


def torch_channelwise_normalize(train_imgs: np.ndarray, val_imgs: np.ndarray, test_imgs: np.ndarray, eps: float = 1e-7) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    train_t = torch.tensor(train_imgs, dtype=torch.float32) / 255.0 # Normalize to [0, 1]
    val_t = torch.tensor(val_imgs, dtype=torch.float32) / 255.0     # Normalize to [0, 1]
    test_t = torch.tensor(test_imgs, dtype=torch.float32) / 255.0   # Normalize to [0, 1]


    # Compute per-channel mean and std over training set (dims: batch, height, width)
    mean = train_t.mean(dim=(0, 2, 3), keepdim=True)
    std = train_t.std(dim=(0, 2, 3), keepdim=True) + eps

    # Apply normalization to all sets using training set's stats
    train_normalized = (train_t - mean) / std
    val_normalized = (val_t - mean) / std
    test_normalized = (test_t - mean) / std

    print("Channel-wise mean (from training data):", mean.view(-1).tolist())
    print("Channel-wise std (from training data): ", std.view(-1).tolist())

    return train_normalized, val_normalized, test_normalized

standard_train_t, standard_validation_t, standard_test_t = \
    torch_channelwise_normalize(train_images_np, validation_images_np, test_images_np)


# --- Data Augmentation and Loading ---

class ImageDataset(torch.utils.data.Dataset):
    """
    A custom PyTorch Dataset for images and their labels.
    Applies optional transformations to images.
    """
    def __init__(self, images: torch.Tensor, labels: torch.Tensor = None, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx: int):
        image = self.images[idx]
        if self.transform:
            image = self.transform(image) # Apply transform

        if self.labels is not None:
            label = self.labels[idx]
            return image, label
        else:
            return image # For test set where labels are not available

# Define data augmentation transformations for training
augmentation_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    # If using torchvision models that expect PIL images, you might need:
    # transforms.ToPILImage(),
    # transforms.RandomRotation(30),
    # transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    # transforms.ToTensor(), # Convert back to tensor if ToPILImage was used
])


train_dataset = ImageDataset(
    standard_train_t,
    train_labels,
    transform=augmentation_transform
)

validation_dataset = ImageDataset(
    standard_validation_t,
    validation_labels
)

# DataLoader for the test set does not need labels
test_dataset = ImageDataset(standard_test_t)


train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
validation_loader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
final_test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

print(f"Train batches: {len(train_loader)}, Validation batches: {len(validation_loader)}, Test batches: {len(final_test_loader)}")


# --- CNN Model Definition ---

class ConvNet(nn.Module):
    """
    A Convolutional Neural Network for image classification.
    Includes Conv2d, BatchNorm2d, ReLU, MaxPool2d, AdaptiveAvgPool2d,
    and Linear layers with Xavier weight initialization.
    """
    def __init__(self, n1=16, n2=32, n3=64, n4=128, n5=256, prediction_classes=NUM_CLASSES):
        super(ConvNet, self).__init__()

        # Convolutional Block 1
        # Input size: 3 x 80 x 80
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=n1, kernel_size=4, padding=1)
        self.normalize1 = nn.BatchNorm2d(n1)
        # Output after conv1: (80 - 4 + 2*1)/1 + 1 = 79 -> (n1, 79, 79)
        # After 2x2 MaxPool: (n1, 39, 39)

        # Convolutional Block 2
        # Input size: n1 x 39 x 39
        self.conv2 = nn.Conv2d(in_channels=n1, out_channels=n2, kernel_size=4, padding=1)
        self.normalize2 = nn.BatchNorm2d(n2)
        # Output after conv2: (39 - 4 + 2*1)/1 + 1 = 38 -> (n2, 38, 38)
        # After 2x2 MaxPool: (n2, 19, 19)

        # Convolutional Block 3
        # Input size: n2 x 19 x 19
        self.conv3 = nn.Conv2d(in_channels=n2, out_channels=n3, kernel_size=5, padding=1)
        self.normalize3 = nn.BatchNorm2d(n3)
        # Output after conv3: (19 - 5 + 2*1)/1 + 1 = 17 -> (n3, 17, 17)
        # After 2x2 MaxPool: (n3, 8, 8)

        # Convolutional Block 4
        # Input size: n3 x 8 x 8
        self.conv4 = nn.Conv2d(in_channels=n3, out_channels=n4, kernel_size=5, padding=1)
        self.normalize4 = nn.BatchNorm2d(n4)
        # Output after conv4: (8 - 5 + 2*1)/1 + 1 = 6 -> (n4, 6, 6)
        # After 2x2 MaxPool: (n4, 3, 3)

        # Global Average Pooling reduces spatial dimensions to 1x1
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        # Output after global_avg_pool: (n4, 1, 1)

        # Fully Connected Layers
        # Flattened input size: n4 * 1 * 1
        self.fc4 = nn.Linear(in_features=n4, out_features=n5)
        self.dropout = nn.Dropout(p=0.5) # Dropout after first FC layer
        self.fc5 = nn.Linear(in_features=n5, out_features=prediction_classes) # Output logits

        # Initialize weights using Xavier Normal method
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1) # Gamma (scale) to 1
                init.constant_(m.bias, 0)   # Beta (shift) to 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.normalize1(self.conv1(x)))
        x = F.max_pool2d(x, kernel_size=2)

        x = F.relu(self.normalize2(self.conv2(x)))
        x = F.max_pool2d(x, kernel_size=2)

        x = F.relu(self.normalize3(self.conv3(x)))
        x = F.max_pool2d(x, kernel_size=2)

        x = F.relu(self.normalize4(self.conv4(x)))
        x = F.max_pool2d(x, kernel_size=2)

        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)

        x = F.relu(self.fc4(x))
        x = self.dropout(x) # Apply dropout after ReLU for fc4
        x = self.fc5(x) # Final output logits
        return x


# --- Model Training Functions ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_one_epoch(dataloader: DataLoader, model: nn.Module, loss_fn: nn.Module, optimizer: optim.Optimizer, scheduler=None, epoch_num: int = 0) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    for batch_idx, (images, labels) in enumerate(dataloader):
        images, labels = images.to(device), labels.to(device)
        labels = labels.long()

        optimizer.zero_grad()
        outputs = model(images).float()
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        correct_predictions += (predicted == labels).sum().item()
        total_samples += labels.size(0)

        if batch_idx % 100 == 0:
            print(f"  Batch {batch_idx * images.size(0):>5d}/{len(dataloader.dataset):>5d}, Loss: {loss.item():>7f}")

        if scheduler and isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
            scheduler.step()

    avg_epoch_loss = total_loss / total_samples
    epoch_accuracy = correct_predictions / total_samples
    return avg_epoch_loss, epoch_accuracy

def validate_one_epoch(dataloader: DataLoader, model: nn.Module, loss_fn: nn.Module) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad(): # Disable gradient calculations during validation
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            labels = labels.long()

            outputs = model(images).float()
            loss = loss_fn(outputs, labels)

            total_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_samples += labels.size(0)

    avg_epoch_loss = total_loss / total_samples
    epoch_accuracy = correct_predictions / total_samples
    return avg_epoch_loss, epoch_accuracy

def get_all_predictions(dataloader: DataLoader, model: nn.Module) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_true_labels = []
    all_predicted_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device) # Labels are needed to compare with predictions

            outputs = model(images).float()
            _, predicted = torch.max(outputs, 1)

            all_true_labels.extend(labels.cpu().numpy())
            all_predicted_labels.extend(predicted.cpu().numpy())

    return np.array(all_true_labels), np.array(all_predicted_labels)


model = ConvNet(prediction_classes=NUM_CLASSES).to(device)
loss_function = nn.CrossEntropyLoss()

optimizer_adam = optim.Adam(model.parameters(), lr=LEARNING_RATE_ADAM, weight_decay=WEIGHT_DECAY_ADAM)
scheduler_adam = torch.optim.lr_scheduler.OneCycleLR(
    optimizer_adam, max_lr=3e-4,
    steps_per_epoch=len(train_loader),
    epochs=NUM_EPOCHS
)

optimizer_sgd = optim.SGD(model.parameters(), lr=LEARNING_RATE_SGD, momentum=MOMENTUM_SGD, weight_decay=WEIGHT_DECAY_SGD)
scheduler_sgd = torch.optim.lr_scheduler.StepLR(
    optimizer_sgd, step_size=10, gamma=0.1
)

# --- Training Loop ---
best_val_loss = float('inf')
patience = 5 # For early stopping, if implemented
patience_counter = 0

# Lists to store metrics for plotting
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []
learning_rates = [] # To track LR changes

print("Starting model training...")
for epoch in range(1, NUM_EPOCHS + 1):
    print(f"--- Epoch {epoch}/{NUM_EPOCHS} ---")

    # Select optimizer and scheduler based on epoch
    if epoch <= SWITCH_OPTIMIZER_EPOCH:
        current_optimizer = optimizer_adam
        current_scheduler = scheduler_adam
    else:
        current_optimizer = optimizer_sgd
        current_scheduler = scheduler_sgd

    # Train for one epoch
    epoch_train_loss, epoch_train_acc = train_one_epoch(
        train_loader, model, loss_function, current_optimizer, current_scheduler, epoch_num=epoch
    )
    train_losses.append(epoch_train_loss)
    train_accuracies.append(epoch_train_acc)

    # Validate for one epoch
    epoch_val_loss, epoch_val_acc = validate_one_epoch(
        validation_loader, model, loss_function
    )
    val_losses.append(epoch_val_loss)
    val_accuracies.append(epoch_val_acc)

    if epoch > SWITCH_OPTIMIZER_EPOCH and current_scheduler:
        current_scheduler.step()

    learning_rates.append(current_optimizer.param_groups[0]['lr'])

    print(f"  Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc*100:.2f}%")
    print(f"  Validation Loss: {epoch_val_loss:.4f}, Validation Acc: {epoch_val_acc*100:.2f}%")
    print(f"  Current Learning Rate: {learning_rates[-1]:.6f}")

print("\nTraining complete!")


epochs_range = range(1, len(train_losses) + 1)

plt.figure(figsize=(14, 6))

# Plot Training and Validation Loss
plt.subplot(1, 2, 1)
plt.plot(epochs_range, train_losses, 'o-', color='blue', label='Training Loss', markersize=4)
plt.plot(epochs_range, val_losses, 'o-', color='red', label='Validation Loss', markersize=4)
plt.title('Training and Validation Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# Plot Training and Validation Accuracy
plt.subplot(1, 2, 2)
plt.plot(epochs_range, [acc * 100 for acc in train_accuracies], 'o-', color='blue', label='Training Accuracy', markersize=4)
plt.plot(epochs_range, [acc * 100 for acc in val_accuracies], 'o-', color='red', label='Validation Accuracy', markersize=4)
plt.title('Training and Validation Accuracy Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Plot Learning Rate Statistics
plt.figure(figsize=(10, 6))
plt.plot(epochs_range, learning_rates, 'o-', color='green', label='Learning Rate', markersize=4)
plt.title('Learning Rate Schedule Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Learning Rate')
plt.legend()
plt.grid(True)
plt.show()


# Confusion Matrix Generation
print("\nGenerating Confusion Matrix for Validation Set...")
final_true_labels, final_predicted_labels = get_all_predictions(validation_loader, model)

cm = confusion_matrix(final_true_labels, final_predicted_labels, labels=range(NUM_CLASSES))

print("Confusion Matrix:")
print(cm)

# Visualize Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=[str(i) for i in range(NUM_CLASSES)], # Use string labels
            yticklabels=[str(i) for i in range(NUM_CLASSES)]) # Use string labels

plt.title('Confusion Matrix (Validation Set)')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()


# --- Run on official tests and create submission CSV ---
model.eval() # Set model to evaluation mode for final predictions
test_predictions = []
test_image_ids = []

with torch.no_grad():
    for images in final_test_loader:
        images = images.to(device)

        outputs = model(images).float()
        _, predicted = torch.max(outputs, 1)

        test_predictions.extend(predicted.cpu().numpy())

        # Collect image IDs for the submission file
        # Note: This assumes final_test_loader maintains order of test_image_ids_map
        # It's better to pass image IDs directly if possible or ensure deterministic loading
        start_idx = len(test_image_ids)
        end_idx = start_idx + images.size(0)
        test_image_ids.extend(list(test_image_ids_map.keys())[start_idx:end_idx])


if len(test_image_ids) != len(test_predictions):
    print("Warning: Mismatch between number of test images and predictions!")
    print(f"Test image IDs collected: {len(test_image_ids)}")
    print(f"Test predictions collected: {len(test_predictions)}")

# Create submission DataFrame
submission_df = pd.DataFrame({
    'image_id': test_image_ids,
    'label': test_predictions
})

submission_df.to_csv('./submission.csv', index=False)
print("Saved predictions to submission.csv")