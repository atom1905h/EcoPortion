import joblib
import numpy as np


def load_models(model_dir="models/", name="", n_models=5):
    models = []
    for i in range(n_models):
        filename = model_dir + f"{name}_{i}.pkl"
        model = joblib.load(filename)
        models.append(model)
    return models


def predict_test_data(models, test_data):
    predictions = []
    for model in models:
        prediction = model.predict(test_data)
        predictions.append(prediction)
    return np.mean(predictions)
