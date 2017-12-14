import os

from sklearn.model_selection import train_test_split

from app.machine_learning_models.CustomRandomForestClassifier import CustomRandomForestClassifier
from app.pipeline.data_vectorizer import DataVectorizer
from app.pipeline.data_loader import DataLoader


def main():
    project_dir = os.path.dirname(os.path.abspath(__file__))
    source_dir = os.path.join(project_dir, '../data/enron_with_categories/1')

    data_loader = DataLoader()
    sorted_categories, sorted_emails = data_loader.load_and_sort_data(source_dir)
    _data, _labels, _filenames = data_loader.get_data_and_labels(sorted_emails, sorted_categories)

    vectorizer = DataVectorizer(_data, _labels)
    data, labels = vectorizer.get_vectorized_data_and_labels()

    X_train, X_val, y_train, y_val = train_test_split(data, labels, random_state=0)

    rfc_model = CustomRandomForestClassifier()
    rfc_model.train(X_train, y_train)



if __name__ == "__main__":
    main()
