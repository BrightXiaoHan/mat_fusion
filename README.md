# 海冰密集度数据融合模型

基于XGBoost的海冰密集度数据融合模型，能够融合多个数据源的海冰观测数据，生成高质量的海冰密集度预测。

## 特性

- **多源数据融合**: 融合Bremen_ASI、NSIDC_CDR、NSIDC_NT2、OSI401、OSI408等5个海冰数据源
- **智能缺失值处理**: 自动处理NaN值（陆地区域），确保数据质量
- **特征重要性分析**: 分析各个数据源的贡献度
- **高效训练**: 基于XGBoost的快速训练，无需GPU
- **模型可解释性**: 提供清晰的特征重要性和预测分析

## 安装依赖

项目使用Python 3.12+，主要依赖包括：

```bash
pip install xgboost scikit-learn joblib h5py matplotlib scipy tqdm click
```

或者如果有uv：

```bash
uv sync
```

## 数据格式

输入数据应为MAT格式，包含以下字段：
- 5个海冰数据源的观测数据
- 对应的坐标信息
- Bremen_MODIS作为目标标签

NaN值表示陆地区域，模型会自动处理这些区域。

## 使用方法

### 1. 数据预处理

将原始MAT数据转换为训练格式：

```bash
python main.py preprocess --input-file assets/sea_ice_dataset_withoutint.mat --output-dir cache/
```

这会生成：
- `cache/train_data.h5`: 训练数据
- `cache/val_data.h5`: 验证数据  
- `cache/test_data.h5`: 测试数据
- `cache/visualization/`: 数据可视化结果

### 2. 模型训练

训练XGBoost模型：

```bash
python main.py train
```

训练完成后生成：
- `checkpoints/xgboost_model.pkl`: 训练好的模型
- `logs/feature_importance.png`: 特征重要性分析图
- `logs/prediction_comparison.png`: 预测效果对比图
- `logs/training_results.txt`: 详细训练结果

### 3. 模型评估

评估模型在测试集上的性能：

```bash
python main.py evaluate
```

生成：
- `cache/evaluation_visualizations/`: 测试样本预测可视化
- `logs/evaluation_results.txt`: 详细评估结果

### 4. 数据可视化

可视化原始海冰数据：

```bash
python main.py visualize --data-file assets/sea_ice_dataset_withoutint.mat
```

## 模型架构

### 特征工程

模型将每个海冰像素转换为7个特征：

1. **Bremen_ASI**: Bremen ASI数据源
2. **NSIDC_CDR**: NSIDC CDR数据源  
3. **NSIDC_NT2**: NSIDC NT2数据源
4. **OSI401**: OSI401数据源
5. **OSI408**: OSI408数据源
6. **row**: 归一化的行坐标（位置信息）
7. **col**: 归一化的列坐标（位置信息）

### XGBoost参数

默认模型参数：
- `max_depth=8`: 树的最大深度
- `learning_rate=0.1`: 学习率
- `n_estimators=2000`: 决策树数量
- `subsample=0.8`: 样本子采样比例
- `colsample_bytree=0.8`: 特征子采样比例
- `early_stopping_rounds=100`: 早停策略

### 缺失值处理策略

- **陆地区域（标签NaN）**: 排除在训练和预测之外
- **输入数据NaN**: 使用其他有效数据源的均值填充
- **全部数据源缺失**: 跳过该像素

## 性能指标

模型使用以下指标评估性能：
- **RMSE** (Root Mean Square Error): 预测误差
- **MSE** (Mean Square Error): 均方误差
- **R²** (决定系数): 模型解释方差的比例

## 项目结构

```
arctic-sea-ice-data-fusion/
├── arctic_sea_ice_data_fusion/          # 主要代码包
│   ├── __init__.py
│   ├── model.py                         # XGBoost模型实现
│   ├── train.py                         # 训练脚本
│   ├── evaluate.py                      # 评估脚本
│   ├── preprocess.py                    # 数据预处理
│   ├── dataset.py                       # 数据集类
│   └── visualize.py                     # 数据可视化
├── assets/                              # 原始数据
├── cache/                               # 预处理后的数据
├── checkpoints/                         # 训练好的模型
├── logs/                               # 训练日志和结果
├── main.py                             # 命令行接口
├── pyproject.toml                      # 项目配置
└── README.md                           # 本文档
```

## API使用示例

### 直接使用模型类

```python
from arctic_sea_ice_data_fusion.model import SeaIceXGBoostModel
import numpy as np

# 初始化模型
model = SeaIceXGBoostModel(
    max_depth=8,
    learning_rate=0.1,
    n_estimators=1000
)

# 准备数据 (N, H, W, 5)
inputs = np.random.rand(100, 64, 64, 5)
labels = np.random.rand(100, 64, 64)
input_masks = np.random.rand(100, 64, 64, 5) > 0.9
label_masks = np.random.rand(100, 64, 64) > 0.8

# 转换为训练格式
X_train, y_train = model.prepare_data(inputs, labels, input_masks, label_masks)

# 训练模型
model.train(X_train, y_train)

# 预测单个图像
predictions, pred_masks = model.predict_image(inputs[:1], input_masks[:1])

# 保存模型
model.save_model('my_model.pkl')
```

### 特征重要性分析

```python
# 获取特征重要性
importance = model.get_feature_importance()
for feature, score in importance.items():
    print(f"{feature}: {score:.4f}")
```

## 输出文件说明

### 训练输出
- `checkpoints/xgboost_model.pkl`: 可以直接加载使用的模型文件
- `logs/feature_importance.png`: 柱状图显示各个特征的重要性
- `logs/prediction_comparison.png`: 散点图对比真实值与预测值
- `logs/training_results.txt`: 包含RMSE、R²等详细指标

### 评估输出  
- `cache/evaluation_visualizations/sample_X.png`: 测试样本的可视化，包括：
  - 5个输入数据源
  - 真实海冰密集度
  - 模型预测结果
  - 预测误差分布

## 常见问题

**Q: 为什么选择XGBoost而不是深度学习模型？**
A: XGBoost在处理表格数据和缺失值方面表现优异，训练速度快，模型可解释性强，特别适合海冰数据的特点。

**Q: 如何处理新的数据？**
A: 确保数据格式与训练数据一致，NaN值标记陆地区域，然后使用`model.predict_image()`进行预测。

**Q: 模型预测速度如何？**
A: 对于单张图像，XGBoost需要逐像素处理，速度可能比CNN慢，但对于批量数据处理效率很高。

**Q: 可以调整模型参数吗？**
A: 可以在训练时传入不同的XGBoost参数，如调整树深度、学习率等来优化性能。

## 引用

如果您在研究中使用了此项目，请引用：

```
Arctic Sea Ice Data Fusion using XGBoost
[您的论文信息]
```

## 许可证

[您的许可证信息]

## 贡献

欢迎提交Issue和Pull Request来改进项目。
