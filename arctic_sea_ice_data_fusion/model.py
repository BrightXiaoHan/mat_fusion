import joblib
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
from tqdm import tqdm


class SeaIceXGBoostModel:
    """
    海冰密集度数据融合的XGBoost模型
    """

    def __init__(self, **xgb_params):
        """
        初始化XGBoost模型

        Args:
            **xgb_params: XGBoost参数
        """
        # 默认参数
        default_params = {
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "max_depth": 6,
            "learning_rate": 0.1,
            "n_estimators": 1000,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "n_jobs": -1,
        }

        # 更新参数
        default_params.update(xgb_params)

        self.model = xgb.XGBRegressor(**default_params)
        self.is_fitted = False
        self.feature_names = None

    def prepare_data(self, inputs, labels, input_masks, label_masks):
        """
        准备训练数据，将图像数据转换为表格数据

        Args:
            inputs: 输入数据 (N, H, W, 5)
            labels: 标签数据 (N, H, W)
            input_masks: 输入掩码 (N, H, W, 5)
            label_masks: 标签掩码 (N, H, W)

        Returns:
            X: 特征矩阵 (n_samples, n_features)
            y: 目标向量 (n_samples,)
        """
        n_samples, height, width, n_channels = inputs.shape

        # 创建特征名称
        if self.feature_names is None:
            channel_names = ["Bremen_ASI", "NSIDC_CDR", "NSIDC_NT2", "OSI401", "OSI408"]
            coord_features = ["row", "col"]  # 添加位置信息作为特征
            self.feature_names = channel_names + coord_features

        X_list = []
        y_list = []

        for sample_idx in tqdm(range(n_samples), desc="准备数据"):
            for row in range(height):
                for col in range(width):
                    # 检查标签是否有效（不是陆地）
                    if not label_masks[sample_idx, row, col] and not np.isnan(
                        labels[sample_idx, row, col]
                    ):
                        # 获取该像素的所有通道数据
                        pixel_features = []
                        valid_pixel = True

                        for ch in range(n_channels):
                            if input_masks[sample_idx, row, col, ch] or np.isnan(
                                inputs[sample_idx, row, col, ch]
                            ):
                                # 如果某个通道的数据无效，用其他通道的均值替代
                                valid_channels = []
                                for other_ch in range(n_channels):
                                    if not input_masks[
                                        sample_idx, row, col, other_ch
                                    ] and not np.isnan(
                                        inputs[sample_idx, row, col, other_ch]
                                    ):
                                        valid_channels.append(
                                            inputs[sample_idx, row, col, other_ch]
                                        )

                                if valid_channels:
                                    pixel_features.append(np.mean(valid_channels))
                                else:
                                    # 如果所有通道都无效，跳过这个像素
                                    valid_pixel = False
                                    break
                            else:
                                pixel_features.append(inputs[sample_idx, row, col, ch])

                        if valid_pixel:
                            # 添加位置信息作为特征
                            pixel_features.extend([row / height, col / width])

                            X_list.append(pixel_features)
                            y_list.append(labels[sample_idx, row, col])

        X = np.array(X_list)
        y = np.array(y_list)

        print(f"准备的数据集: {X.shape[0]} 个有效像素, {X.shape[1]} 个特征")
        return X, y

    def train(self, X_train, y_train, X_val=None, y_val=None, early_stopping_rounds=50):
        """
        训练XGBoost模型

        Args:
            X_train: 训练特征
            y_train: 训练标签
            X_val: 验证特征
            y_val: 验证标签
            early_stopping_rounds: 早停轮数
        """
        # 更新模型参数以包含early_stopping_rounds
        self.model.set_params(early_stopping_rounds=early_stopping_rounds)

        eval_set = [(X_train, y_train)]
        if X_val is not None and y_val is not None:
            eval_set.append((X_val, y_val))

        self.model.fit(X_train, y_train, eval_set=eval_set, verbose=True)

        self.is_fitted = True

    def predict(self, X):
        """
        预测

        Args:
            X: 特征矩阵

        Returns:
            predictions: 预测结果
        """
        if not self.is_fitted:
            raise ValueError("模型尚未训练，请先调用train()方法")

        return self.model.predict(X)

    def predict_image(self, inputs, input_masks):
        """
        对整个图像进行预测

        Args:
            inputs: 输入图像 (N, H, W, 5)
            input_masks: 输入掩码 (N, H, W, 5)

        Returns:
            predictions: 预测图像 (N, H, W)
            prediction_masks: 预测掩码 (N, H, W)
        """
        if not self.is_fitted:
            raise ValueError("模型尚未训练，请先调用train()方法")

        n_samples, height, width, n_channels = inputs.shape
        predictions = np.full((n_samples, height, width), np.nan)
        prediction_masks = np.ones((n_samples, height, width), dtype=bool)

        for sample_idx in tqdm(range(n_samples), desc="预测"):
            for row in range(height):
                for col in range(width):
                    # 准备该像素的特征
                    pixel_features = []
                    valid_pixel = True

                    for ch in range(n_channels):
                        if input_masks[sample_idx, row, col, ch] or np.isnan(
                            inputs[sample_idx, row, col, ch]
                        ):
                            # 处理无效数据
                            valid_channels = []
                            for other_ch in range(n_channels):
                                if not input_masks[
                                    sample_idx, row, col, other_ch
                                ] and not np.isnan(
                                    inputs[sample_idx, row, col, other_ch]
                                ):
                                    valid_channels.append(
                                        inputs[sample_idx, row, col, other_ch]
                                    )

                            if valid_channels:
                                pixel_features.append(np.mean(valid_channels))
                            else:
                                valid_pixel = False
                                break
                        else:
                            pixel_features.append(inputs[sample_idx, row, col, ch])

                    if valid_pixel:
                        # 添加位置信息
                        pixel_features.extend([row / height, col / width])

                        # 预测
                        X_pixel = np.array([pixel_features])
                        pred = self.model.predict(X_pixel)[0]

                        predictions[sample_idx, row, col] = pred
                        prediction_masks[sample_idx, row, col] = False

        return predictions, prediction_masks

    def evaluate(self, X_test, y_test):
        """
        评估模型性能

        Args:
            X_test: 测试特征
            y_test: 测试标签

        Returns:
            metrics: 评估指标字典
        """
        y_pred = self.predict(X_test)

        metrics = {
            "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
            "mse": mean_squared_error(y_test, y_pred),
            "r2": r2_score(y_test, y_pred),
        }

        return metrics

    def save_model(self, filepath):
        """
        保存模型

        Args:
            filepath: 保存路径
        """
        model_data = {
            "model": self.model,
            "is_fitted": self.is_fitted,
            "feature_names": self.feature_names,
        }
        joblib.dump(model_data, filepath)

    def load_model(self, filepath):
        """
        加载模型

        Args:
            filepath: 模型文件路径
        """
        model_data = joblib.load(filepath)
        self.model = model_data["model"]
        self.is_fitted = model_data["is_fitted"]
        self.feature_names = model_data["feature_names"]

    def get_feature_importance(self):
        """
        获取特征重要性

        Returns:
            feature_importance: 特征重要性字典
        """
        if not self.is_fitted:
            raise ValueError("模型尚未训练")

        importance = self.model.feature_importances_
        feature_importance = dict(zip(self.feature_names, importance))

        return feature_importance