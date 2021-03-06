import os

from app.pipeline.enron.data_vectorizer import DataVectorizer
from sklearn.model_selection import train_test_split

from app.constants import INPUT_LENGTH, NO_OF_OUTPUTS
from app.machine_learning_models.LSTMModel import LSTMModel
from app.pipeline.enron.data_loader import DataLoader


def initialize_model():
    project_dir = os.path.dirname(os.path.abspath(__file__))
    source_dir = os.path.join(project_dir, '../data/enron_with_categories/1')

    data_loader = DataLoader(source_dir)
    _data, _labels, _filenames = data_loader.get_data_and_labels()

    vectorizer = DataVectorizer(_data, _labels)
    data = vectorizer.get_vectorized_data(_data)
    labels = vectorizer.get_vectorized_labels()

    X_train, X_val, y_train, y_val = train_test_split(data, labels, random_state=0)

    # model = CustomRandomForestClassifier()
    model = LSTMModel(input_length=INPUT_LENGTH, no_of_outputs=NO_OF_OUTPUTS)
    model.train(X_train, y_train, epochs=1)

    return model, vectorizer


if __name__ == "__main__":
    initialize_model()
