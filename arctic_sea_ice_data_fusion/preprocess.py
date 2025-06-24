# 数据预处理，划分训练集和测试集
import os

import h5py
import numpy as np
from scipy.interpolate import griddata
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from .visualize import visualize_timestep


def preprocess_data(input_file, output_dir):
    """
    预处理海冰数据，包括裁剪、插值和数据集划分

    参数:
        input_file: 输入数据文件路径
        output_dir: 预处理后数据的输出目录
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # Create visualization directory
    visualization_dir = os.path.join(output_dir, "visualization")
    os.makedirs(visualization_dir, exist_ok=True)

    with h5py.File(input_file, "r") as data:
        # 获取Bremen_MODIS的坐标范围作为基准
        modis_x = np.array(data["modis_x"]).flatten()
        modis_y = np.array(data["modis_y"]).flatten()
        modis_extent = (modis_x.min(), modis_x.max(), modis_y.min(), modis_y.max())

        # 创建目标网格（Bremen_MODIS的网格）
        xx, yy = np.meshgrid(modis_x, modis_y)
        target_grid = np.column_stack([xx.ravel(), yy.ravel()])

        # 获取时间步数
        num_timesteps = data["Bremen_MODIS"].shape[0]

        # 数据集映射
        datasets = {
            "Bremen_ASI": ("x_asi", "y_asi"),
            "NSIDC_CDR": ("x_cdr", "y_cdr"),
            "NSIDC_NT2": ("x_nt2", "y_nt2"),
            "OSI401": ("x_osi401", "y_osi401"),
            "OSI408": ("x_osi408", "y_osi408"),
        }

        # Precompute orig_points for each dataset ONCE
        precomputed_points = {}
        for ds_name, (x_key, y_key) in datasets.items():
            x_coords = np.array(data[x_key]).flatten()
            y_coords = np.array(data[y_key]).flatten()
            # Create meshgrid for correct coordinate mapping
            xx, yy = np.meshgrid(x_coords, y_coords)
            precomputed_points[ds_name] = np.column_stack([xx.ravel(), yy.ravel()])

        # Initialize arrays directly instead of appending to lists
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
                
                # Get coordinate dimensions
                x_coords = np.array(data[x_key]).flatten()
                y_coords = np.array(data[y_key]).flatten()
                expected_shape = (len(y_coords), len(x_coords))
                
                # Check and correct data orientation
                if ice_data.shape == expected_shape:
                    # Data is already in correct orientation (lat, lon)
                    corrected_data = ice_data
                elif ice_data.shape == (len(x_coords), len(y_coords)):
                    # Need to transpose to (lat, lon)
                    corrected_data = ice_data.T
                else:
                    print(f"Warning: Shape mismatch for {ds_name} at timestep {t}: "
                          f"data {ice_data.shape}, expected {expected_shape}")
                    corrected_data = ice_data
                
                flat_data = corrected_data.ravel()

                # Use precomputed points
                interp_data = griddata(
                    precomputed_points[ds_name],
                    flat_data,
                    target_grid,
                    method="nearest",
                    fill_value=np.nan,
                ).reshape(len(modis_y), len(modis_x))

                # Record the mask for this channel
                channel_mask = np.isnan(interp_data)
                timestep_inputs_masks.append(channel_mask)
                timestep_inputs.append(interp_data)

            # Stack inputs directly into preallocated array
            all_inputs[t] = np.stack(timestep_inputs, axis=-1)
            # Stack the masks for the inputs for the timestep
            all_inputs_mask[t] = np.stack(timestep_inputs_masks, axis=-1)

            # Handle label
            label_data = np.array(data["Bremen_MODIS"][t])
            if label_data.shape != (len(modis_y), len(modis_x)):
                label_data = label_data.T
            # Invert y-axis for Bremen MODIS
            label_data = np.flipud(label_data)
            all_labels[t] = label_data
            # Record mask for label (where NaNs exist)
            all_labels_mask[t] = np.isnan(all_labels[t])

            # After processing the timestep, visualize it
            visualize_timestep(
                modis_x, 
                modis_y,
                inputs=timestep_inputs,
                label=all_labels[t],
                timestep=t,
                output_dir=visualization_dir
            )

        # 划分数据集 (80%训练, 10%验证, 10%测试)
        # 首先生成索引数组
        indices = np.arange(num_timesteps)

        # 先分出训练集
        train_idx, temp_idx = train_test_split(indices, test_size=0.2, random_state=42)

        # 再将临时集分为验证集和测试集
        val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42)

        # 创建数据集字典
        datasets = {
            "train": {
                "inputs": all_inputs[train_idx],
                "labels": all_labels[train_idx],
                "inputs_mask": all_inputs_mask[train_idx],
                "labels_mask": all_labels_mask[train_idx],
            },
            "val": {
                "inputs": all_inputs[val_idx],
                "labels": all_labels[val_idx],
                "inputs_mask": all_inputs_mask[val_idx],
                "labels_mask": all_labels_mask[val_idx],
            },
            "test": {
                "inputs": all_inputs[test_idx],
                "labels": all_labels[test_idx],
                "inputs_mask": all_inputs_mask[test_idx],
                "labels_mask": all_labels_mask[test_idx],
            },
        }

        # 保存数据集
        for split_name, split_data in datasets.items():
            with h5py.File(os.path.join(output_dir, f"{split_name}_data.h5"), "w") as f:
                f.create_dataset("inputs", data=split_data["inputs"])
                f.create_dataset("labels", data=split_data["labels"])
                f.create_dataset("inputs_mask", data=split_data["inputs_mask"])
                f.create_dataset("labels_mask", data=split_data["labels_mask"])

        print(f"预处理完成! 数据集已保存到: {output_dir}")
        print(f"训练集样本数: {len(train_idx)}")
        print(f"验证集样本数: {len(val_idx)}")
        print(f"测试集样本数: {len(test_idx)}")


if __name__ == "__main__":
    # 配置路径
    input_file = "assets/sea_ice_dataset_withoutint.mat"
    output_dir = "cache/"

    # 运行预处理
    preprocess_data(input_file, output_dir)