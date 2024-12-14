from sklearn.model_selection import train_test_split
import numpy as np
import logging

from AutoML.core import BAGStepwise
import AutoML.optimizer as opt


# подгружаем логгер
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def ModelTrainer(X, y, name):
    """
    Function to create datasets and feature masks based on top features using feature_importance.

    Args:
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Target array.

    Returns:
        list: A list of tuples in the format (OptimizerClass, X_train, y_train, model_name, feature_mask).
    """
    # Выбираем количество признаков
    top_n_features_list = [50, 75, 100, 150]

    # Разделили датасет на ~ 2 части.
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=42)
    print(X_train.shape, X_temp.shape, y_temp.shape)

    # Обучаем модели на разном количестве признаков
    datasets_cb = []
    datasets_rf = []
    datasets_logreg = []
    datasets_xgb = []
    for n_features in top_n_features_list:

        mask = np.zeros(X.shape[1], dtype=int)
        mask[:n_features] = 1

        # Append the dataset information for CatBoost
        model_name = f"{name}_CatBoost_Top_{n_features}_Features"
        logger.info(model_name)
        datasets_cb.append((opt.CatBoostOptimizer,
                           X_train, y_train, model_name, mask))

        model_name = f"{name}_XGB_Top_{n_features}_Features"
        logger.info(model_name)
        datasets_xgb.append((opt.XGB, X_train, y_train,
                            f"XGB_Top{n_features}_Features", mask))

        if n_features >= 100:
            model_name = f"{name}_RandomForest_Top_{n_features}_Features"
            logger.info(model_name)
            datasets_rf.append((opt.randomForest, X_train,
                               y_train, f"RF_Top{n_features}_Features", mask))
        else:

            model_name = f"{name}_LOGREG :)_Top_{n_features}_Features"
            logger.info(model_name)
            datasets_logreg.append(
                (opt.logreg, X_train, y_train, f"logreg_Top{n_features}_Features", mask))

    optimizer = opt.ParallelOptimizer(n_jobs=1)
    models_cb = optimizer.fit(datasets_cb)
    optimizer_alone = opt.ParallelOptimizer(n_jobs=1)
    models_rf = optimizer_alone.fit(datasets_rf)

    optimizer_alone_2 = opt.ParallelOptimizer(n_jobs=1)
    models_logreg = optimizer_alone_2.fit(datasets_logreg)

    optimizer_alone_3 = opt.ParallelOptimizer(n_jobs=1)
    models_xgb = optimizer_alone_3.fit(datasets_xgb)

    models = models_cb + models_rf + models_logreg + models_xgb
    names = [f"{type(model).__name__}{i}" for i, model in enumerate(models)]
    dict_models = {name: model for name, model in zip(names, models)}

    automl = BAGStepwise(models=dict_models, cv=5)
    automl.run(X_temp, y_temp)
    return automl
