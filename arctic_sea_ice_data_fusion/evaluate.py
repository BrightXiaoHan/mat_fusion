import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset import SeaIceDataset
from .model import UNet


def masked_mse(pred, target, mask):
    """Calculate MSE only on unmasked pixels"""
    valid_mask = ~mask
    squared_error = (pred - target) ** 2
    squared_error[~valid_mask] = 0
    num_valid = valid_mask.sum()
    return squared_error.sum() / (num_valid + 1e-8)


def visualize_sample(inputs, target, prediction, sample_idx, output_dir="cache/evaluation_visualizations"):
    """Visualize a single sample with inputs, target, prediction and error"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create figure with 3 rows and 3 columns
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    fig.suptitle(f"Test Sample {sample_idx}", fontsize=16)
    
    # Input channels (5 sources)
    for i in range(5):
        ax = axes[i//3, i%3]
        im = ax.imshow(inputs[i], cmap='Blues', vmin=0, vmax=1)
        ax.set_title(f"Input Channel {i}")
        fig.colorbar(im, ax=ax)
    
    # Target (row 2, col 0)
    ax = axes[2, 0]
    im = ax.imshow(target, cmap='Blues', vmin=0, vmax=1)
    ax.set_title("Target")
    fig.colorbar(im, ax=ax)
    
    # Prediction (row 2, col 1)
    ax = axes[2, 1]
    im = ax.imshow(prediction, cmap='Blues', vmin=0, vmax=1)
    ax.set_title("Prediction")
    fig.colorbar(im, ax=ax)
    
    # Error (row 2, col 2)
    error = np.abs(prediction - target)
    ax = axes[2, 2]
    im = ax.imshow(error, cmap='Reds', vmin=0, vmax=1)
    ax.set_title("Absolute Error")
    fig.colorbar(im, ax=ax)
    
    # Turn off unused axes
    axes[2, 0].axis('off')
    axes[2, 1].axis('off')
    axes[2, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"sample_{sample_idx}.png"), dpi=150)
    plt.close(fig)


def evaluate_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = "cache"
    checkpoint_path = "checkpoints/best_model.pth"

    # Load model
    model = UNet(n_channels=5, n_classes=1).to(device)
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Prepare datasets
    splits = ["train", "val", "test"]
    results = {}

    for split in splits:
        dataset = SeaIceDataset(split, data_dir=data_dir)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

        # Initialize metrics
        input_errors = {f"channel_{i}": [] for i in range(5)}
        pred_errors = []

        sample_idx = 0  # Counter for visualization
        for batch in tqdm(dataloader, desc=f"Evaluating {split}"):
            inputs = batch["input"].to(device)
            labels = batch["label"].to(device)
            input_masks = batch["input_mask"].to(device)
            label_mask = batch["label_mask"].to(device)

            # Model prediction
            with torch.no_grad():
                preds = model(inputs)

            # Calculate prediction error
            pred_error = masked_mse(preds, labels, label_mask)
            pred_errors.append(pred_error.item())

            # Calculate input source errors
            for ch_idx in range(5):
                # Extract single channel
                input_ch = inputs[:, ch_idx : ch_idx + 1]
                input_mask_ch = input_masks[:, ch_idx : ch_idx + 1]

                # Calculate error for this channel
                ch_error = masked_mse(input_ch, labels, label_mask)
                input_errors[f"channel_{ch_idx}"].append(ch_error.item())

            # Visualization for test set
            if split == "test":
                # Convert tensors to numpy arrays
                inputs_np = inputs.cpu().numpy().squeeze(0)  # (5, H, W)
                target_np = labels.cpu().numpy().squeeze()    # (H, W)
                pred_np = preds.cpu().numpy().squeeze()       # (H, W)
                
                # Visualize this sample
                visualize_sample(inputs_np, target_np, pred_np, sample_idx)
                sample_idx += 1

        # Save results for this split
        results[split] = {
            "input_errors": {k: np.mean(v) for k, v in input_errors.items()},
            "prediction_error": np.mean(pred_errors),
        }

    # Print and return results
    for split, metrics in results.items():
        print(f"\n{split.upper()} SET RESULTS:")
        print(f"Prediction Error: {metrics['prediction_error']:.6f}")
        for ch, error in metrics["input_errors"].items():
            print(f"{ch} Error: {error:.6f}")

    return results


if __name__ == "__main__":
    evaluate_model()