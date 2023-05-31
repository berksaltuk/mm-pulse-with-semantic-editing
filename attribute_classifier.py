import torch
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the ResNet model
model = models.resnet18(pretrained=True)
num_classes = 2  # Number of output classes (e.g., smiling or not smiling)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, num_classes)
model = model.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Define data transformations
transform = ToTensor()

# Load and prepare the dataset
train_dataset = ImageFolder('path/to/train/dataset', transform=transform)
test_dataset = ImageFolder('path/to/test/dataset', transform=transform)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)

# Training loop
for epoch in range(10):
    model.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        # Forward pass
        outputs, latent_vectors = model(
            images, return_latent=True)  # Obtain latent vectors

        # Compute loss
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

    # Evaluate the model
    model.eval()
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            # Obtain latent vectors
            _, latent_vectors = model(images, return_latent=True)

            # Perform further analysis with the latent vectors
            # ...
