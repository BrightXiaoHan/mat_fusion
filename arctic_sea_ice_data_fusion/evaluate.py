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
    Visualize image prediction results
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
        fig.suptitle(f"Test Sample {sample_idx}", fontsize=16)
        
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
        ax.set_title("Ground Truth")
        plt.colorbar(im, ax=ax, shrink=0.8)
        
        # 预测结果
        ax = axes[2, 1]
        pred_data = predictions[0]
        pred_mask = prediction_masks[0]
        pred_data_masked = pred_data.copy()
        pred_data_masked[pred_mask] = np.nan
        
        im = ax.imshow(pred_data_masked, cmap='Blues', vmin=0, vmax=1)
        ax.set_title("Prediction")
        plt.colorbar(im, ax=ax, shrink=0.8)
        
        # 误差对比数据（使用右侧中部空白位置）
        ax = axes[1, 2]
        ax.axis('off')  # 关闭坐标轴
        
        # 计算各输入源和预测值相对于真实值的误差
        valid_pixels = ~(label_mask | pred_mask)
        error_data = []
        
        # 计算预测值误差
        if np.any(valid_pixels):
            pred_error = np.sqrt(np.mean((pred_data[valid_pixels] - label_data[valid_pixels]) ** 2))
            error_data.append(('Prediction', pred_error))
        
        # 计算各输入源误差
        for ch_idx, ch_name in enumerate(channel_names):
            channel_data = sample_inputs[0, :, :, ch_idx]
            channel_mask = sample_input_masks[0, :, :, ch_idx]
            
            # 找到该通道和标签都有效的像素
            valid_channel_pixels = ~(label_mask | channel_mask)
            
            if np.any(valid_channel_pixels):
                channel_error = np.sqrt(np.mean((channel_data[valid_channel_pixels] - label_data[valid_channel_pixels]) ** 2))
                error_data.append((ch_name, channel_error))
        
        # 在空白区域显示误差对比表格
        if error_data:
            # 排序误差数据（从小到大）
            error_data.sort(key=lambda x: x[1])
            
            # 创建表格文本
            text_content = "RMSE Comparison:\n" + "="*25 + "\n"
            for source, rmse in error_data:
                text_content += f"{source:<12}: {rmse:.4f}\n"
            
            # 添加相对改善信息
            if len(error_data) > 1:
                best_baseline = min([rmse for source, rmse in error_data if source != 'Prediction'])
                pred_rmse = next(rmse for source, rmse in error_data if source == 'Prediction')
                improvement = ((best_baseline - pred_rmse) / best_baseline) * 100
                text_content += f"\nImprovement: {improvement:+.1f}%"
            
            ax.text(0.05, 0.95, text_content, transform=ax.transAxes, 
                   fontsize=10, verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
        
        # 绝对误差图（移动到右下角）
        ax = axes[2, 2]
        # 只在有效像素处计算误差
        valid_pixels = ~(label_mask | pred_mask)
        error = np.full_like(label_data, np.nan)
        error[valid_pixels] = np.abs(pred_data[valid_pixels] - label_data[valid_pixels])
        
        im = ax.imshow(error, cmap='Reds', vmin=0, vmax=0.5)
        ax.set_title("Absolute Error")
        plt.colorbar(im, ax=ax, shrink=0.8)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"sample_{sample_idx}.png"), dpi=150, bbox_inches='tight')
        plt.close()


def evaluate_model():
    """
    Evaluate XGBoost model
    """
    data_dir = "cache"
    model_path = "checkpoints/xgboost_model.pkl"
    output_dir = "cache/evaluation_visualizations"
    
    # 检查模型文件是否存在
    if not os.path.exists(model_path):
        print(f"Model file does not exist: {model_path}")
        print("Please run training script first to generate model file")
        return
    
    # 加载模型
    print("Loading XGBoost model...")
    model = SeaIceXGBoostModel()
    model.load_model(model_path)
    
    # 加载数据
    print("Loading evaluation data...")
    datasets = load_data_for_evaluation(data_dir)
    
    results = {}
    
    # 评估每个数据集
    for split in ['train', 'val', 'test']:
        print(f"\nEvaluating {split} dataset...")
        
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
        
        print(f"{split} set results:")
        print(f"  RMSE: {metrics['rmse']:.6f}")
        print(f"  MSE: {metrics['mse']:.6f}")
        print(f"  R²: {metrics['r2']:.6f}")
        
        # 计算输入源误差（基线比较）- 优化版本
        print(f"\n{split} set individual input source errors (baseline comparison):")
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
                print(f"  {ch_name}: No valid data")
        
        # 为每个数据集生成可视化
        print(f"\nGenerating {split} sample visualizations...")
        n_samples = min(5, datasets[split]['inputs'].shape[0])
        sample_indices = np.random.choice(datasets[split]['inputs'].shape[0], n_samples, replace=False)
        
        # 为每个数据集创建单独的输出目录
        split_output_dir = os.path.join(output_dir, split)
        
        visualize_image_predictions(
            model,
            datasets[split]['inputs'],
            datasets[split]['labels'],
            datasets[split]['inputs_mask'],
            datasets[split]['labels_mask'],
            sample_indices,
            split_output_dir
        )
        print(f"{split.capitalize()} visualization results saved to: {split_output_dir}")
    
    # 保存评估结果
    results_file = os.path.join("logs", "evaluation_results.txt")
    os.makedirs("logs", exist_ok=True)
    
    with open(results_file, 'w', encoding='utf-8') as f:
        f.write("XGBoost Sea Ice Concentration Data Fusion Model Evaluation Results\n")
        f.write("="*65 + "\n\n")
        
        for split, metrics in results.items():
            f.write(f"{split.upper()} SET:\n")
            f.write(f"  RMSE: {metrics['rmse']:.6f}\n")
            f.write(f"  MSE: {metrics['mse']:.6f}\n")
            f.write(f"  R²: {metrics['r2']:.6f}\n\n")
    
    print(f"\nEvaluation results saved to: {results_file}")
    return results


if __name__ == "__main__":
    evaluate_model()