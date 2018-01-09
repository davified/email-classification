import os
import web

import numpy as np
from sklearn.model_selection import train_test_split
from app.machine_learning_models.CustomRandomForestClassifier import CustomRandomForestClassifier
from app.pipeline.data_vectorizer import DataVectorizer
from app.pipeline.data_loader import DataLoader


class WebServer:
    def GET(self):
        # web.header('Content-Type', 'application/json')
        # user_data = web.input()
        # comment= user_data.message
        model = initialize_model()
        data = get_data()

        return {'predicted classes': model.predict(data)}


def initialize_model():
    project_dir = os.path.dirname(os.path.abspath(__file__))
    source_dir = os.path.join(project_dir, '../data/enron_with_categories/1')

    data_loader = DataLoader(source_dir)
    _data, _labels, _filenames = data_loader.get_data_and_labels()

    vectorizer = DataVectorizer(_data, _labels)
    data = vectorizer.get_vectorized_data()
    labels = vectorizer.get_vectorized_labels()

    X_train, X_val, y_train, y_val = train_test_split(data, labels, random_state=0)

    rfc_model = CustomRandomForestClassifier()
    rfc_model.train(X_train, y_train)
    return rfc_model


def get_data():
    project_dir = os.path.dirname(os.path.abspath(__file__))
    source_dir = os.path.join(project_dir, '../data/enron_with_categories/1')

    data_loader = DataLoader(source_dir)
    _data, _labels, _filenames = data_loader.get_data_and_labels()

    vectorizer = DataVectorizer(_data, _labels)
    data = vectorizer.get_vectorized_data()
    sample_row = data[0,:]
    return np.reshape(sample_row, (-1, 1000))


if __name__ == "__main__":
    urls = ('/', 'WebServer')
    app = web.application(urls, globals())
    app.run()
