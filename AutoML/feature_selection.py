import numpy as np
import pandas as pd
import logging
import threading
from sklearn.ensemble import RandomForestClassifier


# подгружаем логгер
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class FeatureSelector():
    """Селектит фичи"""

    def __init__(self, task: "Task"):
        self.task = task
        self.selected_features = [
            feature for feature in task.df_train.columns if feature.find("feature") != -1]
        self.corr_features = []  # коррелирующие фичи

    def select_features(self):

        self.drop_high_correlated_features()  # убрали корреляцию
        self.RandomForestSelection()  # отсортировали RandomForest

    def drop_high_correlated_features(self, threshold=0.97):
        """Убирает корреляции"""
        try:
            correlation_matrix = self.task.df_train[self.selected_features].corr(
            )
            cor_df = correlation_matrix.to_pandas()

            # Берем только верхнетреугольную матрицу
            upper_triangle = np.triu(np.ones(cor_df.shape), k=1).astype(bool)

            # Выводим скоррелированные пары
            highly_correlated_pairs = [
                (cor_df.columns[i], cor_df.columns[j], cor_df.iloc[i, j])
                for i in range(cor_df.shape[0])
                for j in range(cor_df.shape[1])
                if upper_triangle[i, j] and abs(cor_df.iloc[i, j]) >= threshold
            ]

            # Убираем один из признаков
            self.corr_features = set(pair[1]
                                     for pair in highly_correlated_pairs)
            features = list(set(self.selected_features) - self.corr_features)
            self.selected_features = features

            logger.info(
                f"Количество признаков, оставшихся после корреляции с трешхолдом {threshold}: {len(self.selected_features)}")

        except Exception as e:
            logger.exception(f"drop_high_correlated_features: {e}")

    def _run_random_forest(self, X, y):

        # обучаем RandomForest
        rf = RandomForestClassifier(
            n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X, y)

        # Сортируем признаки в порядке важности модели
        importances = rf.feature_importances_
        selected_features_with_importance = [
            (feature, importance) for feature, importance in zip(self.selected_features, importances)
        ]
        selected_features_with_importance.sort(
            key=lambda x: x[1], reverse=True)

        self.selected_features = [feature for feature,
                                  _ in selected_features_with_importance]
        logger.info("Отбор признаков завершен")

    def RandomForestSelection(self, timeout=600):
        try:
            # Подготовка данных
            X = self.task.df_train.select(self.selected_features).to_numpy()
            y = self.task.df_train.select("target").to_numpy().ravel()

            # Проверка на пустые массивы
            if X.shape[0] == 0 or X.shape[1] == 0:
                raise ValueError("Массив признаков пуст после фильтрации.")

            # Создаем поток для выполнения отбора признаков
            thread = threading.Thread(
                target=self._run_random_forest, args=(X, y))
            thread.start()
            thread.join(timeout=timeout)

            if thread.is_alive():
                raise TimeoutError(
                    f"RandomForest exceeded the time limit of {timeout} seconds.")

        except TimeoutError as e:
            logger.error(e)
        except Exception as e:
            logger.exception(
                f"An error occurred during RandomForest execution: {e}")
