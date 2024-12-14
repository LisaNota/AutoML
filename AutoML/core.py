import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import cross_val_score, KFold
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression

import numpy as np

def custom_cross_val_score(model, X, y, scoring=roc_auc_score, cv=5):
    """
    Выполняет кросс-валидацию для кастомных моделей.

    Parameters:
    - model: объект модели с методами `fit` и `predict`.
    - X: ndarray, матрица признаков.
    - y: ndarray, вектор целевых переменных.
    - cv: int, количество фолдов (по умолчанию 5).

    Returns:
    - scores: ndarray, метрики на каждом фолде.
    """
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    scores = []
    for train_idx, val_idx in kf.split(X):
        # Разделение данных
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        # Обучение модели
        model.fit(X_train, y_train)

        # Предсказание
        y_pred = model.predict_proba(X_val)[:,-1]
        # Оценка качества
        score = scoring(y_val, y_pred)
        scores.append(score)

    return np.array(scores)


class CustomStackingModel:
    """
    Кастомная модель стекинга для классификации.
    Поддерживает произвольные модели, если у них есть методы `fit` и `predict`.
    """

    def __init__(self, base_models, meta_model, use_probas=True):
        """
        Parameters:
        - base_models: list, список базовых моделей.
        - meta_model: модель мета-уровня.
        - use_probas: bool, если True, базовые модели должны поддерживать `predict_proba`.
        """
        self.base_models = base_models
        self.meta_model = meta_model
        self.use_probas = use_probas

    def fit(self, X, y):
        """
        Обучение базовых моделей и мета-модели.
        """
        # Обучение базовых моделей
        base_predictions = []
        for name, model in self.base_models:
            if self.use_probas:
                base_predictions.append(model.predict_proba(X).reshape((-1,1)))
            else:
                base_predictions.append(model.predict(X).reshape(-1, 1))

        # Генерация входов для мета-модели
        meta_features = np.hstack(base_predictions)
        self.meta_model.fit(meta_features, y)

    def predict(self, X):
        """
        Предсказание мета-модели на новых данных.
        """
        base_predictions = []
        for name, model in self.base_models:
            if self.use_probas:
                base_predictions.append(model.predict_proba(X).reshape((-1,1)))
            else:
                base_predictions.append(model.predict(X).reshape(-1, 1))

        meta_features = np.hstack(base_predictions)
        return self.meta_model.predict(meta_features)

    def predict_proba(self, X):
        """
        Предсказание вероятностей (если поддерживается мета-моделью).
        """
        base_predictions = []
        for name, model in self.base_models:
            if self.use_probas:
                base_predictions.append(model.predict_proba(X).reshape((-1,1)))
            else:
                raise ValueError("Для `predict_proba` базовые модели должны поддерживать `predict_proba`.")

        meta_features = np.hstack(base_predictions)
        return self.meta_model.predict_proba(meta_features)


# Stepwise-алгоритм с forward и backward проходами
class StepwiseStackingSelector:
    def __init__(self, models, meta_model=None, cv=5):
        self.models = models  # Доступные модели
        self.meta_model = meta_model or LogisticRegression()  # Мета-модель для стекинга
        self.cv = cv
        self.selected_models = []  # Список выбранных моделей (имя, модель, оценка)

    def fit(self, X, y):
        remaining_models = list(self.models.items())  # Модели, которые еще не добавлены
        current_ensemble = []  # Текущий ансамбль
        best_score = -np.inf  # Текущая лучшая метрика

        while remaining_models or len(current_ensemble) > 1:  # Пока есть что добавлять или убирать
            improvement = False

            # Forward шаг: пробуем добавить новую модель
            if remaining_models:
                best_forward_model = None
                best_forward_model_name = None

                for model_name, model in remaining_models:
                    temp_ensemble = current_ensemble + [(model_name, model)]
                    stacking_model = self._build_stacking_model(temp_ensemble)
                    score = custom_cross_val_score(stacking_model, X, y, cv=self.cv).mean()

                    print(f"Trying to add: {model_name}, Score: {score:.4f}")
                    if score > best_score:
                        best_score = score
                        best_forward_model = model
                        best_forward_model_name = model_name
                        improvement = True

                if best_forward_model:
                    current_ensemble.append((best_forward_model_name, best_forward_model))
                    self.selected_models.append((best_forward_model_name, best_forward_model))
                    remaining_models = list(filter(lambda x: x[0] != best_forward_model_name, remaining_models))
                    print(f"Added model: {best_forward_model_name}, New Score: {best_score:.4f}")

            # Backward шаг: пробуем удалить существующие модели
            if len(current_ensemble) > 1:
                for i, (model_name, model) in enumerate(current_ensemble):
                    temp_ensemble = current_ensemble[:i] + current_ensemble[i + 1:]
                    stacking_model = self._build_stacking_model(temp_ensemble)
                    score = custom_cross_val_score(stacking_model, X, y, cv=self.cv).mean()

                    print(f"Trying to remove: {model_name}, Score: {score:.4f}")
                    if score > best_score:
                        best_score = score
                        removed_model = current_ensemble.pop(i)
                        self.selected_models = [(name, mdl) for name, mdl in self.selected_models if
                                                name != removed_model[0]]
                        improvement = True
                        print(f"Removed model: {model_name}, New Score: {best_score:.4f}")
                        break  # Проверяем следующий backward шаг

            if not improvement:
                print("No further improvement possible. Stopping...")
                break  # Если ни forward, ни backward не дали улучшений

    def _build_stacking_model(self, ensemble):
        # return StackingClassifier(
        #     estimators=ensemble,
        #     final_estimator=self.meta_model,
        #     passthrough=False
        # )
        return CustomStackingModel(
            base_models=ensemble,
            meta_model=self.meta_model
        )

    def get_final_estimator(self, X, y):
        final_model = self._build_stacking_model(self.selected_models)
        final_model.fit(X, y)
        return self.selected_models, final_model


class BAGStepwise:
    def __init__(self, models, cv=5):
        self.models = models
        self.cv = cv
        self.selected_models = None
        self.ensemble = None

    def run(self, X, y):
        # Stepwise отбор моделей

        selector = StepwiseStackingSelector(models=self.models, cv=self.cv)
        selector.fit(X, y)
        self.selected_models, self.ensemble = selector.get_final_estimator(X, y)
        return self.ensemble

    def predict(self, X):
        if not self.ensemble:
            raise ValueError("Сначала выполните обучение (run)!")
        return self.ensemble.predict(X)

    def predict_proba(self, X):
        if not self.ensemble:
            raise ValueError("Сначала выполните обучение (run)!")
        return self.ensemble.predict_proba(X)[:, 1]

    def score(self, X, y):
        if not self.ensemble:
            raise ValueError("Сначала выполните обучение (run)!")
        return accuracy_score(y, self.ensemble.predict(X))




# Пример использования
# if __name__ == "__main__":
#     from sklearn.datasets import load_iris
#     from sklearn.linear_model import LogisticRegression
#     from sklearn.svm import SVC
#     from sklearn.tree import DecisionTreeClassifier
#     from sklearn.ensemble import RandomForestClassifier
#     from sklearn.model_selection import train_test_split
#
#     # Данные
#     data = load_iris()
#     X, y = data.data, data.target
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
#     # Модели для AutoML
#     models = {
#         "RandomForest": RandomForestClassifier(n_estimators=100),
#         "LogisticRegression": LogisticRegression(max_iter=1000),
#         "DecisionTree": DecisionTreeClassifier(),
#         "SVM": SVC(probability=True)
#     }
#
#     # AutoML
#     automl = BAGStepwise(models=models, scoring='accuracy', cv=5)
#     automl.run(X_train, y_train)
#
#     print("Stacking Score on Test Set:", automl.score(X_test, y_test))
