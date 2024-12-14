
import numpy as np
from catboost import CatBoostClassifier
from xgboost import XGBClassifier

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


class ModelOptimizer:
    """Базовый класс для оптимизации гиперпараметров моделей."""

    def __init__(self, random_state=42):
        self.random_state = random_state
        self.best_params = None
        self.model = None
        self.feature_mask = None

    def objective(self, trial, X_train, X_val, y_train, y_val):
        """Определяет функцию оптимизации (переопределяется в наследниках)."""
        raise NotImplementedError

    def train_model(self, X, y, timeout=15*60, feature_mask=None):
        """
        Обучает модель с оптимизированными гиперпараметрами.

        Parameters:
        - X: numpy.ndarray, матрица признаков.
        - y: numpy.ndarray, целевая переменная.
        - n_trials: int, количество итераций Optuna.
        - feature_mask: list or numpy.ndarray, маска признаков [0, 1].
        """
        # Применяем feature_mask, если он задан
        if feature_mask is not None:
            X = self._apply_feature_mask(X, feature_mask)
            self.feature_mask = feature_mask

        self.best_params = {
            "iterations": 1000,
            "learning_rate": 0.1,
            # "task_type": "GPU",
            "depth": 6,
            "loss_function": "Logloss",
            "verbose": 100,
            "random_seed": self.random_state
        }

        # Сохранение лучших гиперпараметров и обучение модели
        self.model = self._build_model(self.best_params)
        self.model.fit(X, y)
        return self

    def predict_proba(self, X):
        if self.feature_mask is not None:
            X = self._apply_feature_mask(X, self.feature_mask)
        y_pred = self.model.predict_proba(X)[:, 1]
        return y_pred

    def predict(self, X):
        if self.feature_mask is not None:
            X = self._apply_feature_mask(X, self.feature_mask)
        y_pred = self.model.predict(X)
        return y_pred

    def _apply_feature_mask(self, X, feature_mask):
        """
        Применяет маску к признакам.

        :param X: numpy.ndarray, матрица признаков.
        :param feature_mask: list or numpy.ndarray, маска [0, 1].
        :return: numpy.ndarray, матрица признаков с учетом маски.
        """
        if len(feature_mask) != X.shape[1]:
            raise ValueError(
                "Размер feature_mask должен совпадать с числом признаков X.")
        mask = np.array(feature_mask, dtype=bool)
        return X[:, mask]

    def _build_model(self, params):
        """Создаёт модель с заданными параметрами (переопределяется в наследниках)."""
        raise NotImplementedError


class CatBoostOptimizer(ModelOptimizer):
    def _build_model(self, params):
        return CatBoostClassifier(**params)


class XGB(ModelOptimizer):

    def train_model(self, X, y, feature_mask=None):
        """
        Обучает модель с оптимизированными гиперпараметрами.

        Parameters:
        - X: numpy.ndarray, матрица признаков.
        - y: numpy.ndarray, целевая переменная.
        - n_trials: int, количество итераций Optuna.
        - feature_mask: list or numpy.ndarray, маска признаков [0, 1].
        """
        # Применяем feature_mask, если он задан
        if feature_mask is not None:
            X = self._apply_feature_mask(X, feature_mask)
            self.feature_mask = feature_mask

        self.best_params = {
            "objective": "binary:logistic",  # Задача бинарной классификации
            "eval_metric": "auc",           # Метрика ROC-AUC
            "eta": 0.1,                     # Скорость обучения
            "max_depth": 6,                 # Максимальная глубина дерева
            "n_estimators": 500,            # Количество деревьев
            "random_state": self.random_state,  # Случайное состояние для воспроизводимости
            "tree_method": "gpu_hist",      # Использование GPU для ускорения
        }

        # Сохранение лучших гиперпараметров и обучение модели
        self.model = self._build_model(self.best_params)
        self.model.fit(X, y)
        return self

    def _build_model(self, params):
        return XGBClassifier(**params)


class logreg(ModelOptimizer):

    def train_model(self, X, y, feature_mask=None):
        """
        Обучает модель с оптимизированными гиперпараметрами.

        Parameters:
        - X: numpy.ndarray, матрица признаков.
        - y: numpy.ndarray, целевая переменная.
        - n_trials: int, количество итераций Optuna.
        - feature_mask: list or numpy.ndarray, маска признаков [0, 1].
        """
        # Применяем feature_mask, если он задан
        if feature_mask is not None:
            X = self._apply_feature_mask(X, feature_mask)
            self.feature_mask = feature_mask

        self.best_params = {
            # Регуляризация (можно 'l1', 'l2', 'elasticnet', или 'none')
            "penalty": "l2",
            # Инверсия коэффициента регуляризации (чем меньше, тем сильнее регуляризация)
            "C": 1.0,
            # Оптимизатор ('liblinear', 'saga', 'lbfgs', и др.)
            "solver": "liblinear",
            "max_iter": 1000,              # Максимальное количество итераций
            "class_weight": "balanced",    # Сбалансировка классов
            # Установка случайного состояния для воспроизводимости
            "random_state": self.random_state
        }
        # Сохранение лучших гиперпараметров и обучение модели
        self.model = self._build_model(self.best_params)
        self.model.fit(X, y)
        return self

    def _build_model(self, params):
        return LogisticRegression(**params)


class randomForest(ModelOptimizer):

    def train_model(self, X, y, feature_mask=None):
        """
        Обучает модель с оптимизированными гиперпараметрами.

        Parameters:
        - X: numpy.ndarray, матрица признаков.
        - y: numpy.ndarray, целевая переменная.
        - n_trials: int, количество итераций Optuna.
        - feature_mask: list or numpy.ndarray, маска признаков [0, 1].
        """
        # Применяем feature_mask, если он задан
        if feature_mask is not None:
            X = self._apply_feature_mask(X, feature_mask)
            self.feature_mask = feature_mask

        self.best_params = {
            "n_estimators": 500,
            "criterion": "entropy",
            "max_depth": 6,
            "max_features": "sqrt",
            "verbose": 1,
            "random_state": self.random_state,
            "n_jobs": -1
        }

        # Сохранение лучших гиперпараметров и обучение модели
        self.model = self._build_model(self.best_params)
        self.model.fit(X, y)
        return self

    def _build_model(self, params):
        return RandomForestClassifier(**params)
