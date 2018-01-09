import os

from sklearn.model_selection import train_test_split

from app.machine_learning_models.CustomRandomForestClassifier import CustomRandomForestClassifier
from app.machine_learning_models.LSTMModel import LSTMModel
from app.pipeline.data_vectorizer import DataVectorizer
from app.pipeline.data_loader import DataLoader


def main():
    INPUT_LENGTH = 1000
    NO_OF_OUTPUTS = 17

    project_dir = os.path.dirname(os.path.abspath(__file__))
    source_dir = os.path.join(project_dir, '../data/enron_with_categories/1')

    data_loader = DataLoader(source_dir)
    _data, _labels, _filenames = data_loader.get_data_and_labels()

    vectorizer = DataVectorizer(_data, _labels)
    data, labels = vectorizer.get_vectorized_data_and_labels()

    X_train, X_val, y_train, y_val = train_test_split(data, labels, random_state=0)

    # model = CustomRandomForestClassifier()
    model = LSTMModel(input_length=INPUT_LENGTH, no_of_outputs=NO_OF_OUTPUTS)
    model.train(X_train, y_train)


if __name__ == "__main__":
    main()
