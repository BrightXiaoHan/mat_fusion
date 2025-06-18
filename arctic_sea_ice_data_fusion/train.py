import os

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .dataset import SeaIceDataset
from .model import UNet


def masked_mse_loss(pred, target, mask):
    """
    Calculate MSE loss only on unmasked pixels

    Args:
        pred: predicted tensor (B, 1, H, W)
        target: ground truth tensor (B, 1, H, W)
        mask: boolean mask tensor (B, 1, H, W) where True indicates invalid pixel
    """
    # Invert mask: True -> valid pixel, False -> invalid
    valid_mask = ~mask

    # Calculate squared error only on valid pixels
    squared_error = (pred - target) ** 2
    squared_error[~valid_mask] = 0  # Zero out invalid pixels

    # Count valid pixels per sample
    num_valid = valid_mask.sum(dim=(1, 2, 3))

    # Calculate loss per sample and average
    loss_per_sample = squared_error.sum(dim=(1, 2, 3)) / (num_valid + 1e-8)
    return loss_per_sample.mean()


def train_model():
    # Configuration
    data_dir = "cache"
    log_dir = "logs"
    checkpoint_dir = "checkpoints"
    num_epochs = 100
    learning_rate = 1e-4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create directories
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Initialize model, loss, and optimizer
    model = UNet(n_channels=5, n_classes=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Create datasets and dataloaders with batch_size=1
    train_dataset = SeaIceDataset("train", data_dir=data_dir)
    val_dataset = SeaIceDataset("val", data_dir=data_dir)

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=2)

    # Tensorboard writer
    writer = SummaryWriter(log_dir)

    # Training variables
    best_val_loss = float("inf")
    start_epoch = 0

    # Check for existing checkpoint
    checkpoint_path = os.path.join(checkpoint_dir, "best_model.pth")
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        best_val_loss = checkpoint["best_val_loss"]
        start_epoch = checkpoint["epoch"] + 1
        print(f"Resuming training from epoch {start_epoch}")

    # Training loop
    for epoch in range(start_epoch, num_epochs):
        model.train()
        epoch_train_loss = 0.0
        train_step = 0

        # Training phase
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Train]"):
            # Move data to device
            inputs = batch["input"].to(device)
            labels = batch["label"].to(device)
            label_masks = batch["label_mask"].to(device)

            # Forward pass
            outputs = model(inputs)

            # Calculate loss
            loss = masked_mse_loss(outputs, labels, label_masks)

            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Accumulate loss
            epoch_train_loss += loss.item()
            train_step += 1

        # Calculate average training loss
        avg_train_loss = epoch_train_loss / train_step
        writer.add_scalar("Loss/train", avg_train_loss, epoch)

        # Validation phase
        model.eval()
        epoch_val_loss = 0.0
        val_step = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Val]"):
                inputs = batch["input"].to(device)
                labels = batch["label"].to(device)
                label_masks = batch["label_mask"].to(device)

                outputs = model(inputs)
                loss = masked_mse_loss(outputs, labels, label_masks)

                epoch_val_loss += loss.item()
                val_step += 1

        avg_val_loss = epoch_val_loss / val_step
        writer.add_scalar("Loss/val", avg_val_loss, epoch)

        print(
            f"Epoch {epoch + 1}/{num_epochs} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}"
        )

        # Save checkpoint if validation loss improves
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_loss": best_val_loss,
                },
                checkpoint_path,
            )
            print(f"Saved new best model with val loss: {best_val_loss:.6f}")

    print("Training complete!")
    writer.close()


if __name__ == "__main__":
    train_model()