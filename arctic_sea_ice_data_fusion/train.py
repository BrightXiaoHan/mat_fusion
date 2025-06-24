import os
import numpy as np
import h5py
from tqdm import tqdm
import matplotlib.pyplot as plt

from .dataset import SeaIceDataset
from .model import SeaIceXGBoostModel


def load_data_for_xgboost(data_dir):
    """
    为XGBoost加载和准备数据

    Args:
        data_dir: 数据目录
        
    Returns:
        训练、验证和测试数据
    """
    # 加载数据集
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


def train_model():
    """
    训练XGBoost模型进行海冰密集度数据融合
    """
    # 配置
    data_dir = "cache"
    checkpoint_dir = "checkpoints"
    log_dir = "logs"

    # 创建目录
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    print("正在加载数据集...")
    datasets = load_data_for_xgboost(data_dir)
    
    # 初始化模型
    model = SeaIceXGBoostModel(
        max_depth=8,
        learning_rate=0.1,
        n_estimators=2000,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    
    print("正在准备训练数据...")
    X_train, y_train = model.prepare_data(
        datasets['train']['inputs'],
        datasets['train']['labels'],
        datasets['train']['inputs_mask'],
        datasets['train']['labels_mask']
    )
    
    print("正在准备验证数据...")
    X_val, y_val = model.prepare_data(
        datasets['val']['inputs'],
        datasets['val']['labels'],
        datasets['val']['inputs_mask'],
        datasets['val']['labels_mask']
    )
    
    print("正在准备测试数据...")
    X_test, y_test = model.prepare_data(
        datasets['test']['inputs'],
        datasets['test']['labels'],
        datasets['test']['inputs_mask'],
        datasets['test']['labels_mask']
    )
    
    print(f"训练集: {X_train.shape[0]} 个样本")
    print(f"验证集: {X_val.shape[0]} 个样本")
    print(f"测试集: {X_test.shape[0]} 个样本")
    
    # 训练模型
    print("开始训练XGBoost模型...")
    model.train(X_train, y_train, X_val, y_val, early_stopping_rounds=100)
    
    # 评估模型
    print("评估训练集性能...")
    train_metrics = model.evaluate(X_train, y_train)
    print(f"训练集 - RMSE: {train_metrics['rmse']:.4f}, R²: {train_metrics['r2']:.4f}")
    
    print("评估验证集性能...")
    val_metrics = model.evaluate(X_val, y_val)
    print(f"验证集 - RMSE: {val_metrics['rmse']:.4f}, R²: {val_metrics['r2']:.4f}")
    
    print("评估测试集性能...")
    test_metrics = model.evaluate(X_test, y_test)
    print(f"测试集 - RMSE: {test_metrics['rmse']:.4f}, R²: {test_metrics['r2']:.4f}")
    
    # 保存模型
    model_path = os.path.join(checkpoint_dir, "xgboost_model.pkl")
    model.save_model(model_path)
    print(f"模型已保存到: {model_path}")
    
    # 获取并打印特征重要性
    feature_importance = model.get_feature_importance()
    print("\n特征重要性:")
    for feature, importance in sorted(feature_importance.items(), key=lambda x: x[1], reverse=True):
        print(f"{feature}: {importance:.4f}")
    
    # 绘制特征重要性图
    plt.figure(figsize=(10, 6))
    features = list(feature_importance.keys())
    importances = [feature_importance[f] for f in features]
    
    plt.barh(features, importances)
    plt.xlabel('重要性')
    plt.title('XGBoost特征重要性')
    plt.tight_layout()
    
    importance_plot_path = os.path.join(log_dir, "feature_importance.png")
    plt.savefig(importance_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"特征重要性图已保存到: {importance_plot_path}")
    
    # 绘制真实值vs预测值散点图
    plt.figure(figsize=(12, 4))
    
    # 训练集
    plt.subplot(1, 3, 1)
    y_train_pred = model.predict(X_train)
    plt.scatter(y_train, y_train_pred, alpha=0.5, s=1)
    plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=2)
    plt.xlabel('真实值')
    plt.ylabel('预测值')
    plt.title(f'训练集 (R²={train_metrics["r2"]:.3f})')
    
    # 验证集
    plt.subplot(1, 3, 2)
    y_val_pred = model.predict(X_val)
    plt.scatter(y_val, y_val_pred, alpha=0.5, s=1)
    plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--', lw=2)
    plt.xlabel('真实值')
    plt.ylabel('预测值')
    plt.title(f'验证集 (R²={val_metrics["r2"]:.3f})')
    
    # 测试集
    plt.subplot(1, 3, 3)
    y_test_pred = model.predict(X_test)
    plt.scatter(y_test, y_test_pred, alpha=0.5, s=1)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('真实值')
    plt.ylabel('预测值')
    plt.title(f'测试集 (R²={test_metrics["r2"]:.3f})')
    
    plt.tight_layout()
    prediction_plot_path = os.path.join(log_dir, "prediction_comparison.png")
    plt.savefig(prediction_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"预测对比图已保存到: {prediction_plot_path}")
    
    # 保存评估结果
    results = {
        'train': train_metrics,
        'val': val_metrics,
        'test': test_metrics,
        'feature_importance': feature_importance
    }
    
    results_path = os.path.join(log_dir, "training_results.txt")
    with open(results_path, 'w', encoding='utf-8') as f:
        f.write("XGBoost海冰密集度数据融合模型训练结果\n")
        f.write("=" * 50 + "\n\n")
        
        for split, metrics in [('训练集', train_metrics), ('验证集', val_metrics), ('测试集', test_metrics)]:
            f.write(f"{split}:\n")
            f.write(f"  RMSE: {metrics['rmse']:.6f}\n")
            f.write(f"  MSE: {metrics['mse']:.6f}\n")
            f.write(f"  R²: {metrics['r2']:.6f}\n\n")
        
        f.write("特征重要性:\n")
        for feature, importance in sorted(feature_importance.items(), key=lambda x: x[1], reverse=True):
            f.write(f"  {feature}: {importance:.6f}\n")
    
    print(f"训练结果已保存到: {results_path}")
    print("训练完成!")


if __name__ == "__main__":
    train_model()