import os

from sklearn.model_selection import train_test_split
from keras.models import model_from_json

from app.pipeline.yelp.data_loader import DataLoader
from app.pipeline.yelp.data_vectorizer import DataVectorizer
from app.constants import INPUT_LENGTH, NO_OF_OUTPUTS
from app.machine_learning_models.LSTMModel import LSTMModel


def initialize_model(retrain_model=False):
    app_dir = os.path.dirname(os.path.abspath(__file__))
    source_dir = os.path.join(app_dir, '../data/yelp/dataset')
    file_path = os.path.join(source_dir, 'review_5_samples.json')
    models_cache_dir = os.path.join(app_dir, './models_cache')
    data_loader = DataLoader()

    _data, _labels, _ = data_loader.get_data_and_labels(file_path)
    vectorizer = DataVectorizer(_data, _labels, num_words=20000)

    if retrain_model or len(os.listdir(models_cache_dir)) < 2:
        data = vectorizer.get_vectorized_data(_data, max_length=INPUT_LENGTH)
        labels = _labels

        X_train, X_val, y_train, y_val = train_test_split(data, labels, random_state=0)

        model = LSTMModel(input_length=INPUT_LENGTH, no_of_outputs=NO_OF_OUTPUTS)
        model.train(X_train, y_train, epochs=1)

        save_model_and_weights(model.classifier, models_cache_dir)

    else:
        model = load_model_and_weights(models_cache_dir)

    return model, vectorizer


def save_model_and_weights(model, models_cache_dir):
    model_json = model.to_json()
    model_filepath = os.path.join(models_cache_dir, 'model.json')
    weights_filepath = os.path.join(models_cache_dir, 'model.h5')

    with open(model_filepath, 'w+') as json_file:
        json_file.write(model_json)

    model.save_weights(weights_filepath)


def load_model_and_weights(models_cache_dir):
    model_filepath = os.path.join(models_cache_dir, 'model.json')
    weights_filepath = os.path.join(models_cache_dir, 'model.h5')
    json_file = open(model_filepath, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(weights_filepath)

    return loaded_model


if __name__ == '__main__':
    # initialize_model(retrain_model=True)
    initialize_model()
