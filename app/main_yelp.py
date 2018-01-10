import os

from sklearn.model_selection import train_test_split

from app.pipeline.yelp.data_loader import DataLoader
from app.pipeline.yelp.data_vectorizer import DataVectorizer
from app.constants import INPUT_LENGTH, NO_OF_OUTPUTS
from app.machine_learning_models.LSTMModel import LSTMModel


def initialize_model():
    project_dir = os.path.dirname(os.path.abspath(__file__))
    source_dir = os.path.join(project_dir, '../data/yelp/dataset')
    file_path = os.path.join(source_dir, 'review_downsized.json')
    data_loader = DataLoader()

    _data, _labels, _ = data_loader.get_data_and_labels(file_path)

    vectorizer = DataVectorizer(_data, _labels, num_words=20000)
    data = vectorizer.get_vectorized_data(_data, max_length=INPUT_LENGTH)
    # labels = vectorizer.get_vectorized_labels()
    labels = _labels

    X_train, X_val, y_train, y_val = train_test_split(data, labels, random_state=0)

    # model = CustomRandomForestClassifier()
    model = LSTMModel(input_length=INPUT_LENGTH, no_of_outputs=NO_OF_OUTPUTS)
    model.train(X_train, y_train, epochs=1)

    return model, vectorizer


if __name__ == "__main__":
    initialize_model()
