"""
Age regression example using the IMDB-WIKI dataset.

This example demonstrates how to use torchregress with a pretrained model
for age regression on the IMDB-WIKI dataset.
"""

import os
import zipfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split

from torchregress.losses import (
    LogCoshLoss,
    QuantileLoss,
    WeightedLossWrapper,
)
from torchregress.metrics.point import mae, mse, rmse

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Define constants
DATA_DIR = os.path.join("data", "imdb_wiki")
IMDB_WIKI_URL = "https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/wiki_crop.zip"
BATCH_SIZE = 32
NUM_EPOCHS = 20
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def download_and_extract_dataset():
    """Download and extract a subset of the IMDB-WIKI dataset (Wiki part)."""
    os.makedirs(DATA_DIR, exist_ok=True)

    zip_path = os.path.join(DATA_DIR, "wiki_crop.zip")

    # Check if the data directory already has extracted files
    if (
        os.path.exists(os.path.join(DATA_DIR, "wiki_crop"))
        and len(os.listdir(os.path.join(DATA_DIR, "wiki_crop"))) > 0
    ):
        print("Dataset already downloaded and extracted.")
        return

    # Download the dataset if not already downloaded
    if not os.path.exists(zip_path):
        print(f"Downloading Wiki dataset from {IMDB_WIKI_URL}...")
        print("This may take a while as the file is ~1GB...")

        # Stream download with progress
        with requests.get(IMDB_WIKI_URL, stream=True, timeout=30) as r:
            r.raise_for_status()
            total_size = int(r.headers.get("content-length", 0))

            with open(zip_path, "wb") as f:
                downloaded = 0
                for chunk in r.iter_content(chunk_size=8192):
                    downloaded += len(chunk)
                    f.write(chunk)

                    # Print progress
                    done = int(50 * downloaded / total_size)
                    downloaded_mb = downloaded / 1024 / 1024
                    total_mb = total_size / 1024 / 1024
                    progress_bar = f"\r[{'=' * done}{' ' * (50 - done)}]"
                    print(
                        f"{progress_bar} {downloaded_mb:.1f}/{total_mb:.1f} MB",
                        end="",
                        flush=True,
                    )
        print("\nDownload completed.")

    # Extract the dataset
    if not os.path.exists(os.path.join(DATA_DIR, "wiki_crop")):
        print("Extracting dataset...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            # Prevent Zip Slip vulnerability by validating paths
            target_dir = os.path.abspath(DATA_DIR)
            for member in zip_ref.namelist():
                member_path = os.path.abspath(os.path.join(target_dir, member))
                if os.path.commonpath([target_dir, member_path]) != target_dir:
                    raise Exception("Attempted Path Traversal in Zip File")
            zip_ref.extractall(DATA_DIR)
        print("Extraction completed.")


def create_metadata():
    """Create a metadata file for the Wiki dataset with filtered entries."""
    meta_file = os.path.join(DATA_DIR, "wiki_metadata.csv")

    # Check if metadata already exists
    if os.path.exists(meta_file):
        print("Metadata file already exists.")
        return pd.read_csv(meta_file)

    import scipy.io

    # Load the MATLAB metadata file
    wiki_mat = scipy.io.loadmat(os.path.join(DATA_DIR, "wiki_crop", "wiki.mat"))
    wiki_data = wiki_mat["wiki"][0][0]

    # Extract metadata fields
    full_paths = [os.path.join(DATA_DIR, "wiki_crop", p[0]) for p in wiki_data[2][0]]
    face_scores = wiki_data[6][0]
    second_face_scores = wiki_data[7][0]
    ages = wiki_data[1][0]

    # Create a DataFrame
    df = pd.DataFrame(
        {
            "path": full_paths,
            "age": ages,
            "face_score": face_scores,
            "second_face_score": second_face_scores,
        }
    )

    # Filter out invalid entries:
    # 1. Keep ages between 0-100
    # 2. Keep images with high face detection confidence
    # 3. Remove images with multiple faces
    filtered_df = df[
        (df["age"] >= 0)
        & (df["age"] <= 100)
        & (df["face_score"] > 1.0)
        & (np.isnan(df["second_face_score"]))
    ]

    # Save to CSV
    print(f"Saving {len(filtered_df)} valid entries to metadata file...")
    filtered_df.to_csv(meta_file, index=False)
    return filtered_df


class IMDBWIKIDataset(Dataset):
    """IMDB-WIKI dataset for age regression."""

    def __init__(self, metadata, transform=None):
        self.metadata = metadata
        self.transform = transform

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        img_path = self.metadata.iloc[idx]["path"]
        age = self.metadata.iloc[idx]["age"]

        # Load and convert image
        try:
            image = Image.open(img_path).convert("RGB")

            if self.transform:
                image = self.transform(image)

            return image, torch.tensor([age], dtype=torch.float32)
        except Exception as e:
            # If there's an issue with the image, return a default one
            print(f"Error loading image {img_path}: {e}")
            # Return a blank image and the mean age
            blank_image = torch.zeros(3, 224, 224)
            return blank_image, torch.tensor([20.0], dtype=torch.float32)


class AgeRegressionModel(nn.Module):
    """Age regression model using a pretrained backbone."""

    def __init__(self, pretrained=True):
        super().__init__()
        # Use ResNet18 as the backbone
        resnet = models.resnet18(pretrained=pretrained)

        # Remove the final classification layer
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])

        # Add regression head
        self.regression_head = nn.Sequential(
            nn.Flatten(), nn.Linear(512, 128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, 1)
        )

    def forward(self, x):
        features = self.backbone(x)
        age_pred = self.regression_head(features)
        return age_pred


def train_model(model, train_loader, val_loader, loss_fn, optimizer, num_epochs=5):
    """Train the age regression model."""
    model.to(DEVICE)

    best_val_loss = float("inf")
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_loss)

        # Validation phase
        model.eval()
        val_running_loss = 0.0

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                outputs = model(inputs)
                loss = loss_fn(outputs, targets)
                val_running_loss += loss.item() * inputs.size(0)

        val_epoch_loss = val_running_loss / len(val_loader.dataset)
        val_losses.append(val_epoch_loss)

        message = (
            f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {epoch_loss:.4f}, "
            f"Val Loss: {val_epoch_loss:.4f}"
        )
        print(message)

        # Save the best model
        if val_epoch_loss < best_val_loss:
            best_val_loss = val_epoch_loss
            # Optionally save model here

    return model, train_losses, val_losses


def evaluate_model(model, test_loader):
    """Evaluate the model on the test set."""
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            outputs = model(inputs)

            all_preds.append(outputs.cpu())
            all_targets.append(targets.cpu())

    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    # Calculate metrics
    mae_value = mae(all_preds, all_targets)
    mse_value = mse(all_preds, all_targets)
    rmse_value = rmse(all_preds, all_targets)

    return {
        "mae": mae_value.item(),
        "mse": mse_value.item(),
        "rmse": rmse_value.item(),
        "predictions": all_preds.numpy(),
        "targets": all_targets.numpy(),
    }


def main():
    print(f"Using device: {DEVICE}")

    # Download and prepare dataset
    try:
        download_and_extract_dataset()
        metadata_df = create_metadata()
    except Exception as e:
        print(f"Error setting up the dataset: {e}")
        print("Please check the dataset URL and try again.")
        return

    # Create train/val/test split - using a smaller subset for demonstration
    # This makes the example run faster by using only 2000 samples
    sample_size = min(2000, len(metadata_df))
    metadata_sample = metadata_df.sample(sample_size, random_state=42).reset_index(drop=True)

    # Define transforms
    transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Create dataset
    full_dataset = IMDBWIKIDataset(metadata_sample, transform=transform)

    # Split dataset
    train_size = int(0.7 * len(full_dataset))
    val_size = int(0.15 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(42)
    )

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    # Define different loss functions to compare
    loss_functions = {
        "MSE": WeightedLossWrapper(nn.MSELoss, reduction="mean"),
        "MAE": WeightedLossWrapper(nn.L1Loss, reduction="mean"),
        "Huber": WeightedLossWrapper(nn.HuberLoss, reduction="mean"),
        "LogCosh": LogCoshLoss(reduction="mean"),  # Missing reduction parameter
        "Quantile_50": QuantileLoss(quantile=0.5, reduction="mean"),  # Missing reduction parameter
    }

    # Dictionary to store results
    results = {}

    # Train models with different loss functions
    for loss_name, loss_fn in loss_functions.items():
        print(f"\n=== Training with {loss_name} Loss ===")

        # Initialize model
        model = AgeRegressionModel(pretrained=True)

        # Initialize optimizer
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Train model
        model, train_losses, val_losses = train_model(
            model, train_loader, val_loader, loss_fn, optimizer, num_epochs=NUM_EPOCHS
        )

        # Evaluate model
        metrics = evaluate_model(model, test_loader)

        # Store results
        results[loss_name] = {
            "model": model,
            "train_losses": train_losses,
            "val_losses": val_losses,
            "metrics": metrics,
        }

    # Compare the results
    plt.figure(figsize=(20, 15))

    # Plot training losses
    plt.subplot(2, 2, 1)
    for loss_name, result in results.items():
        plt.plot(result["train_losses"], label=f"{loss_name}")
    plt.title("Training Loss Curves")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    # Plot validation losses
    plt.subplot(2, 2, 2)
    for loss_name, result in results.items():
        plt.plot(result["val_losses"], label=f"{loss_name}")
    plt.title("Validation Loss Curves")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    # Plot predicted vs actual ages
    plt.subplot(2, 2, 3)
    for loss_name, result in results.items():
        metrics = result["metrics"]
        plt.scatter(
            metrics["targets"][:100, 0],  # Plot first 100 points for clarity
            metrics["predictions"][:100, 0],
            alpha=0.5,
            label=f"{loss_name} (MAE: {metrics['mae']:.2f})",
        )

    # Add identity line
    min_val = min(
        [np.min(results[loss_name]["metrics"]["targets"]) for loss_name in loss_functions]
    )
    max_val = max(
        [np.max(results[loss_name]["metrics"]["targets"]) for loss_name in loss_functions]
    )
    plt.plot([min_val, max_val], [min_val, max_val], "k--")
    plt.title("Predicted vs Actual Age")
    plt.xlabel("Actual Age")
    plt.ylabel("Predicted Age")
    plt.legend()

    # Plot error distribution
    plt.subplot(2, 2, 4)
    for loss_name, result in results.items():
        metrics = result["metrics"]
        errors = metrics["predictions"] - metrics["targets"]
        plt.hist(errors.flatten(), bins=30, alpha=0.5, label=f"{loss_name}")
    plt.title("Error Distribution")
    plt.xlabel("Prediction Error (years)")
    plt.ylabel("Frequency")
    plt.legend()

    plt.tight_layout()
    plt.savefig("imdb_wiki_age_regression_results.png")
    plt.show()

    # Print results table
    print("\nResults Summary:")
    print("-" * 60)
    print(f"{'Loss Function':<15} {'MAE':<10} {'MSE':<10} {'RMSE':<10}")
    print("-" * 60)
    for loss_name, result in results.items():
        metrics = result["metrics"]
        print(
            f"{loss_name:<15} {metrics['mae']:<10.2f} "
            f"{metrics['mse']:<10.2f} {metrics['rmse']:<10.2f}"
        )


if __name__ == "__main__":
    main()
