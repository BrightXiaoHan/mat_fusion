# Inference script for the Arctic Sea Ice Data Fusion model

import argparse
import h5py
import numpy as np
from tqdm import tqdm

# Inference script for the Arctic Sea Ice Data Fusion model
import h5py
import numpy as np
from scipy.interpolate import griddata
from tqdm import tqdm
import argparse
from .model import SeaIceXGBoostModel

def preprocess_for_inference(input_file):
    """
    Preprocess the input .mat file for inference.
    
    Args:
        input_file: Path to the input .mat file.
    
    Returns:
        Tuple containing:
        - inputs: Array of shape [T, H, W, 5] containing the 5 input channels
        - labels: Array of shape [T, H, W] containing the ground truth
        - inputs_mask: Boolean array of shape [T, H, W, 5] indicating missing data
        - labels_mask: Boolean array of shape [T, H, W] indicating missing labels
    """
    with h5py.File(input_file, "r") as data:
        # Get the Bremen_MODIS grid
        modis_x = np.array(data["modis_x"]).flatten()
        modis_y = np.array(data["modis_y"]).flatten()
        
        # Create target grid
        xx, yy = np.meshgrid(modis_x, modis_y)
        target_grid = np.column_stack([xx.ravel(), yy.ravel()])
        
        num_timesteps = data["Bremen_MODIS"].shape[0]
        
        # Define datasets and their coordinate keys
        datasets = {
            "Bremen_ASI": ("x_asi", "y_asi"),
            "NSIDC_CDR": ("x_cdr", "y_cdr"),
            "NSIDC_NT2": ("x_nt2", "y_nt2"),
            "OSI401": ("x_osi401", "y_osi401"),
            "OSI408": ("x_osi408", "y_osi408"),
        }
        
        # Precompute coordinate points for each dataset
        precomputed_points = {}
        for ds_name, (x_key, y_key) in datasets.items():
            x_coords = np.array(data[x_key]).flatten()
            y_coords = np.array(data[y_key]).flatten()
            xx, yy = np.meshgrid(x_coords, y_coords)
            precomputed_points[ds_name] = np.column_stack([xx.ravel(), yy.ravel()])
        
        # Initialize arrays
        all_inputs = np.empty((num_timesteps, len(modis_y), len(modis_x), 5))
        all_labels = np.empty((num_timesteps, len(modis_y), len(modis_x)))
        all_inputs_mask = np.empty((num_timesteps, len(modis_y), len(modis_x), 5), dtype=bool)
        all_labels_mask = np.empty((num_timesteps, len(modis_y), len(modis_x)), dtype=bool)
        
        # Process each timestep
        for t in tqdm(range(num_timesteps)):
            timestep_inputs = []
            timestep_inputs_masks = []
            
            for ds_name, (x_key, y_key) in datasets.items():
                ice_data = np.array(data[ds_name][t])
                x_coords = np.array(data[x_key]).flatten()
                y_coords = np.array(data[y_key]).flatten()
                expected_shape = (len(y_coords), len(x_coords))
                
                # Check and correct data orientation
                if ice_data.shape == expected_shape:
                    corrected_data = ice_data
                elif ice_data.shape == (len(x_coords), len(y_coords)):
                    corrected_data = ice_data.T
                else:
                    print(f"Warning: Shape mismatch for {ds_name} at timestep {t}")
                    corrected_data = ice_data
                
                flat_data = corrected_data.ravel()
                
                # Interpolate to target grid
                interp_data = griddata(
                    precomputed_points[ds_name],
                    flat_data,
                    target_grid,
                    method="nearest",
                    fill_value=np.nan,
                ).reshape(len(modis_y), len(modis_x))
                
                # Record mask for this channel
                channel_mask = np.isnan(interp_data)
                timestep_inputs_masks.append(channel_mask)
                timestep_inputs.append(interp_data)
            
            # Stack inputs and masks for current timestep
            all_inputs[t] = np.stack(timestep_inputs, axis=-1)
            all_inputs_mask[t] = np.stack(timestep_inputs_masks, axis=-1)
            
            # Process label (Bremen_MODIS)
            label_data = np.array(data["Bremen_MODIS"][t])
            if label_data.shape != (len(modis_y), len(modis_x)):
                label_data = label_data.T
            label_data = np.flipud(label_data)  # Invert y-axis
            all_labels[t] = label_data
            all_labels_mask[t] = np.isnan(all_labels[t])
    
    return all_inputs, all_labels, all_inputs_mask, all_labels_mask


def compute_metrics(predictions, labels, labels_mask):
    """
    Compute evaluation metrics for predictions.
    
    Args:
        predictions: Array of shape [T, H, W] of model predictions
        labels: Array of shape [T, H, W] of ground truth values
        labels_mask: Boolean array of shape [T, H, W] indicating missing labels
        
    Returns:
        Dictionary containing:
        - rmse: Root Mean Squared Error
        - mse: Mean Squared Error
        - r2: R-squared coefficient
        - mae: Mean Absolute Error
    """
    # Create combined mask for valid pixels
    valid_mask = ~labels_mask & ~np.isnan(labels) & ~np.isnan(predictions)
    
    # Flatten arrays and apply mask
    y_true = labels[valid_mask]
    y_pred = predictions[valid_mask]
    
    if len(y_true) == 0:
        return {
            "rmse": np.nan,
            "mse": np.nan,
            "r2": np.nan,
            "mae": np.nan
        }
    
    # Calculate metrics
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_true - y_pred))
    
    # Calculate R²
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else np.nan
    
    return {
        "rmse": rmse,
        "mse": mse,
        "r2": r2,
        "mae": mae
    }

def run_inference(input_file, output_file, model_path="checkpoints/xgboost_model.pkl"):
    """
    Run inference on a .mat file and save results to a new .mat file
    
    Args:
        input_file: Path to input .mat file
        output_file: Path to output .mat file
        model_path: Path to trained model file
    """
    # Preprocess input data
    print("Preprocessing input data...")
    inputs, labels, inputs_mask, labels_mask = preprocess_for_inference(input_file)
    
    # Load trained model
    print("Loading model...")
    model = SeaIceXGBoostModel()
    model.load_model(model_path)
    
    # Initialize arrays for predictions and errors
    num_timesteps = inputs.shape[0]
    height, width = inputs.shape[1], inputs.shape[2]
    predictions = np.empty((num_timesteps, height, width))
    prediction_masks = np.empty((num_timesteps, height, width), dtype=bool)
    
    # Initialize arrays for input errors
    input_errors = np.empty((num_timesteps, height, width, 5))
    input_errors[:] = np.nan
    
    # Run inference for each timestep
    print("Running inference...")
    for t in tqdm(range(num_timesteps)):
        # Get current timestep data
        timestep_inputs = inputs[t:t+1]
        timestep_inputs_mask = inputs_mask[t:t+1]
        
        # Predict using model
        pred, pred_mask = model.predict_image(timestep_inputs, timestep_inputs_mask)
        predictions[t] = pred[0]
        prediction_masks[t] = pred_mask[0]
        
        # Calculate input errors
        for ch in range(5):
            # Only calculate where both input and label are valid
            valid_pixels = ~inputs_mask[t, :, :, ch] & ~labels_mask[t] & ~np.isnan(labels[t])
            if np.any(valid_pixels):
                input_errors[t, :, :, ch] = np.abs(inputs[t, :, :, ch] - labels[t])
    
    # Calculate model metrics
    print("Calculating metrics...")
    model_metrics = compute_metrics(predictions, labels, labels_mask)
    
    # Calculate input metrics
    input_metrics = {}
    channel_names = ['Bremen_ASI', 'NSIDC_CDR', 'NSIDC_NT2', 'OSI401', 'OSI408']
    for ch, name in enumerate(channel_names):
        # Flatten arrays and apply valid mask
        valid_mask = ~inputs_mask[:, :, :, ch] & ~labels_mask & ~np.isnan(labels)
        if np.any(valid_mask):
            ch_errors = np.abs(inputs[:, :, :, ch][valid_mask] - labels[valid_mask])
            rmse = np.sqrt(np.mean(ch_errors ** 2))
            mae = np.mean(ch_errors)
            input_metrics[name] = {"rmse": rmse, "mae": mae}
        else:
            input_metrics[name] = {"rmse": np.nan, "mae": np.nan}
    
    # Save results to .mat file
    print(f"Saving results to {output_file}...")
    with h5py.File(output_file, "w") as f:
        # Save data arrays
        f.create_dataset("inputs", data=inputs)
        f.create_dataset("labels", data=labels)
        f.create_dataset("predictions", data=predictions)
        f.create_dataset("input_errors", data=input_errors)
        
        # Save masks
        f.create_dataset("inputs_mask", data=inputs_mask)
        f.create_dataset("labels_mask", data=labels_mask)
        f.create_dataset("predictions_mask", data=prediction_masks)
        
        # Save metrics
        metrics_group = f.create_group("metrics")
        model_metrics_group = metrics_group.create_group("model")
        for k, v in model_metrics.items():
            model_metrics_group.attrs[k] = v
        
        input_metrics_group = metrics_group.create_group("inputs")
        for name, metrics in input_metrics.items():
            ch_group = input_metrics_group.create_group(name)
            for k, v in metrics.items():
                ch_group.attrs[k] = v
    
    print("Inference complete!")
    print("Model Metrics:")
    for k, v in model_metrics.items():
        print(f"  {k}: {v:.6f}")
    
    print("\nInput Metrics:")
    for name, metrics in input_metrics.items():
        print(f"  {name}:")
        for k, v in metrics.items():
            print(f"    {k}: {v:.6f}")

if __name__ == "__main__":
    # Set up command line arguments
    parser = argparse.ArgumentParser(description="Run inference on Arctic sea ice data")
    parser.add_argument("--input", type=str, required=True,
                        help="Path to input .mat file")
    parser.add_argument("--output", type=str, required=True,
                        help="Path to output .mat file")
    parser.add_argument("--model", type=str, default="checkpoints/xgboost_model.pkl",
                        help="Path to trained model file")
    
    args = parser.parse_args()
    
    # Run inference
    run_inference(args.input, args.output, args.model)