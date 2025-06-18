# 数据集
import os
import h5py
import torch
from torch.utils.data import Dataset

class SeaIceDataset(Dataset):
    def __init__(self, split, data_dir="cache", transform=None):
        """
        Dataset for sea ice concentration data
        
        Args:
            split (str): 'train', 'val', or 'test'
            data_dir (str): Directory containing preprocessed HDF5 files
            transform (callable, optional): Optional transforms to apply
        """
        self.split = split
        self.data_dir = data_dir
        self.transform = transform
        
        # Load data from HDF5 file
        file_path = os.path.join(data_dir, f"{split}_data.h5")
        with h5py.File(file_path, "r") as f:
            self.inputs = f["inputs"][:]  # Shape: (N, H, W, 5)
            self.labels = f["labels"][:]  # Shape: (N, H, W)
            self.inputs_mask = f["inputs_mask"][:]  # Shape: (N, H, W, 5)
            self.labels_mask = f["labels_mask"][:]  # Shape: (N, H, W)
    
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        # Get data for this index
        input_data = self.inputs[idx]  # (H, W, 5)
        label = self.labels[idx]       # (H, W)
        input_mask = self.inputs_mask[idx]  # (H, W, 5)
        label_mask = self.labels_mask[idx]   # (H, W)
        
        # Convert to PyTorch tensors and adjust dimensions
        # Move channel dimension to first position for inputs
        input_tensor = torch.tensor(input_data, dtype=torch.float32).permute(2, 0, 1)
        label_tensor = torch.tensor(label, dtype=torch.float32).unsqueeze(0)  # Add channel dim
        input_mask_tensor = torch.tensor(input_mask, dtype=torch.bool).permute(2, 0, 1)
        label_mask_tensor = torch.tensor(label_mask, dtype=torch.bool).unsqueeze(0)
        
        sample = {
            "input": input_tensor,
            "label": label_tensor,
            "input_mask": input_mask_tensor,
            "label_mask": label_mask_tensor
        }
        
        if self.transform:
            sample = self.transform(sample)
            
        return sample