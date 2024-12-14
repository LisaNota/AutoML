from AutoML import FeatureSelector, ModelTrainer
import polars as pl
from dotenv import load_dotenv
import os
import logging


load_dotenv()

PATH_TO_DATA = os.getenv("PATH_TO_DATA")

# подгружаем логгер
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class Pipeline():
    """Основной пайплайн для решения задачи"""

    def __init__(self):
        self.data_folder = PATH_TO_DATA
        self.tasks = []  # массив с задачами класса Task

    def run(self):
        """Собсна сам пайплайн"""
        self.add_tasks()  # создаем задачу под один датасет
        for task in self.tasks:

            logger.info(f"Начинается отбор признаков для {task.name}")
            task.load_train_test_dataset()  # подгрузили наборы данных

            # заполнение пропусков (заглушка; явно указано, что пропусков в данных не будет)
            task.df_train = task.df_train.fill_null(0)
            task.df_test = task.df_test.fill_null(0)

            # отбор признаков
            selector_feat = FeatureSelector(task)
            selector_feat.select_features()

            X = task.df_train.select(
                selector_feat.selected_features).to_numpy()
            y = task.df_train.select('target').to_numpy()

            # обучение и тюнинг моделей
            super_model = ModelTrainer(X, y, task.name)

            test_array = task.df_test.select(
                selector_feat.selected_features).to_numpy()
            predictions = super_model.predict_proba(test_array)

            # формирование предсказаний, сохранение данных
            predictions_series = pl.Series("target", predictions)
            df_test = task.df_test.with_columns(predictions_series)

            # Преобразование типа id, если требуется
            df_test = df_test.with_columns(
                pl.col("id")  # Если "id" должен быть строкой
            )

            df_test = df_test.select(["id", "target"]).sort("id")

            # создание папки если ее нет
            output_dir = "predictions"
            os.makedirs(output_dir, exist_ok=True)

            df_test.write_csv(f"predictions/{task.name}.csv")

            logger.info(f"Fitted model: {task.name}. And saved results")

            del task.df_train
            del task.df_test


if __name__ == '__main__':
    pipe = Pipeline()
    pipe.run()
