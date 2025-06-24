import os
import matplotlib.pyplot as plt
import numpy as np
import h5py
from tqdm import tqdm

from .model import SeaIceXGBoostModel


def load_data_for_evaluation(data_dir):
    """
    加载评估数据
    """
    datasets = {}
    for split in ['train', 'val', 'test']:
        file_path = os.path.join(data_dir, f"{split}_data.h5")
        with h5py.File(file_path, "r") as f:
            datasets[split] = {
                'inputs': f["inputs"][:],
                'labels': f["labels"][:],
                'inputs_mask': f["inputs_mask"][:],
                'labels_mask': f["labels_mask"][:]
            }
    return datasets


def visualize_image_predictions(model, inputs, labels, input_masks, label_masks, sample_indices, output_dir):
    """
    可视化图像预测结果
    """
    os.makedirs(output_dir, exist_ok=True)
    
    channel_names = ['Bremen_ASI', 'NSIDC_CDR', 'NSIDC_NT2', 'OSI401', 'OSI408']
    
    for i, sample_idx in enumerate(sample_indices):
        # 获取该样本的预测结果
        sample_inputs = inputs[sample_idx:sample_idx+1]
        sample_labels = labels[sample_idx:sample_idx+1]
        sample_input_masks = input_masks[sample_idx:sample_idx+1]
        sample_label_masks = label_masks[sample_idx:sample_idx+1]
        
        # 生成预测
        predictions, prediction_masks = model.predict_image(sample_inputs, sample_input_masks)
        
        # 创建可视化
        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        fig.suptitle(f"测试样本 {sample_idx}", fontsize=16)
        
        # 输入通道（5个）
        for ch in range(5):
            ax = axes[ch//3, ch%3]
            data = sample_inputs[0, :, :, ch]
            mask = sample_input_masks[0, :, :, ch]
            
            # 将掩码区域设为NaN以便可视化
            data_masked = data.copy()
            data_masked[mask] = np.nan
            
            im = ax.imshow(data_masked, cmap='Blues', vmin=0, vmax=1)
            ax.set_title(f"{channel_names[ch]}")
            plt.colorbar(im, ax=ax, shrink=0.8)
        
        # 真实标签
        ax = axes[2, 0]
        label_data = sample_labels[0]
        label_mask = sample_label_masks[0]
        label_data_masked = label_data.copy()
        label_data_masked[label_mask] = np.nan
        
        im = ax.imshow(label_data_masked, cmap='Blues', vmin=0, vmax=1)
        ax.set_title("真实值")
        plt.colorbar(im, ax=ax, shrink=0.8)
        
        # 预测结果
        ax = axes[2, 1]
        pred_data = predictions[0]
        pred_mask = prediction_masks[0]
        pred_data_masked = pred_data.copy()
        pred_data_masked[pred_mask] = np.nan
        
        im = ax.imshow(pred_data_masked, cmap='Blues', vmin=0, vmax=1)
        ax.set_title("预测值")
        plt.colorbar(im, ax=ax, shrink=0.8)
        
        # 误差
        ax = axes[2, 2]
        # 只在有效像素处计算误差
        valid_pixels = ~(label_mask | pred_mask)
        error = np.full_like(label_data, np.nan)
        error[valid_pixels] = np.abs(pred_data[valid_pixels] - label_data[valid_pixels])
        
        im = ax.imshow(error, cmap='Reds', vmin=0, vmax=0.5)
        ax.set_title("绝对误差")
        plt.colorbar(im, ax=ax, shrink=0.8)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"sample_{sample_idx}.png"), dpi=150, bbox_inches='tight')
        plt.close()


def evaluate_model():
    """
    评估XGBoost模型
    """
    data_dir = "cache"
    model_path = "checkpoints/xgboost_model.pkl"
    output_dir = "cache/evaluation_visualizations"
    
    # 检查模型文件是否存在
    if not os.path.exists(model_path):
        print(f"模型文件不存在: {model_path}")
        print("请先运行训练脚本生成模型文件")
        return
    
    # 加载模型
    print("加载XGBoost模型...")
    model = SeaIceXGBoostModel()
    model.load_model(model_path)
    
    # 加载数据
    print("加载评估数据...")
    datasets = load_data_for_evaluation(data_dir)
    
    results = {}
    
    # 评估每个数据集
    for split in ['train', 'val', 'test']:
        print(f"\n评估 {split} 数据集...")
        
        # 准备数据
        X, y = model.prepare_data(
            datasets[split]['inputs'],
            datasets[split]['labels'],
            datasets[split]['inputs_mask'],
            datasets[split]['labels_mask']
        )
        
        # 评估性能
        metrics = model.evaluate(X, y)
        results[split] = metrics
        
        print(f"{split} 集结果:")
        print(f"  RMSE: {metrics['rmse']:.6f}")
        print(f"  MSE: {metrics['mse']:.6f}")
        print(f"  R²: {metrics['r2']:.6f}")
        
        # 计算输入源误差（基线比较）- 优化版本
        print(f"\n{split} 集各输入源误差 (作为基线比较):")
        channel_names = ['Bremen_ASI', 'NSIDC_CDR', 'NSIDC_NT2', 'OSI401', 'OSI408']
        
        # 使用向量化操作优化性能
        labels_data = datasets[split]['labels']
        labels_mask = datasets[split]['labels_mask']
        inputs_data = datasets[split]['inputs']
        inputs_mask = datasets[split]['inputs_mask']
        
        # 创建有效标签的掩码
        valid_labels = ~labels_mask & ~np.isnan(labels_data)
        
        for ch_idx, ch_name in enumerate(channel_names):
            # 获取该通道的数据和掩码
            channel_data = inputs_data[:, :, :, ch_idx]
            channel_mask = inputs_mask[:, :, :, ch_idx]
            
            # 创建该通道有效数据的掩码
            valid_channel = ~channel_mask & ~np.isnan(channel_data)
            
            # 找到同时有效的像素（标签和该通道都有效）
            valid_both = valid_labels & valid_channel
            
            if np.any(valid_both):
                # 提取有效的预测值和目标值
                channel_predictions = channel_data[valid_both]
                channel_targets = labels_data[valid_both]
                
                # 计算该通道的RMSE
                channel_rmse = np.sqrt(np.mean((channel_predictions - channel_targets) ** 2))
                print(f"  {ch_name}: RMSE = {channel_rmse:.6f}")
            else:
                print(f"  {ch_name}: 无有效数据")
    
    # 可视化测试集中的几个样本
    if 'test' in datasets:
        print("\n生成测试样本可视化...")
        n_samples = min(5, datasets['test']['inputs'].shape[0])
        sample_indices = np.random.choice(datasets['test']['inputs'].shape[0], n_samples, replace=False)
        
        visualize_image_predictions(
            model,
            datasets['test']['inputs'],
            datasets['test']['labels'],
            datasets['test']['inputs_mask'],
            datasets['test']['labels_mask'],
            sample_indices,
            output_dir
        )
        print(f"可视化结果已保存到: {output_dir}")
    
    # 保存评估结果
    results_file = os.path.join("logs", "evaluation_results.txt")
    os.makedirs("logs", exist_ok=True)
    
    with open(results_file, 'w', encoding='utf-8') as f:
        f.write("XGBoost海冰密集度数据融合模型评估结果\n")
        f.write("="*50 + "\n\n")
        
        for split, metrics in results.items():
            f.write(f"{split.upper()} 集:\n")
            f.write(f"  RMSE: {metrics['rmse']:.6f}\n")
            f.write(f"  MSE: {metrics['mse']:.6f}\n")
            f.write(f"  R²: {metrics['r2']:.6f}\n\n")
    
    print(f"\n评估结果已保存到: {results_file}")
    return results


if __name__ == "__main__":
    evaluate_model()