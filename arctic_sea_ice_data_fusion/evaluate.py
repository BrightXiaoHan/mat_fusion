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