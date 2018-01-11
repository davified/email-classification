import os, pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from app.pipeline.yelp.data_loader import DataLoader
from app.pipeline.yelp.data_vectorizer import DataVectorizer


def initialize_model(retrain_model=False):
    app_dir = os.path.dirname(os.path.abspath(__file__))
    source_dir = os.path.join(app_dir, '../data/yelp/dataset')
    models_cache_dir = os.path.join(app_dir, './models_cache')
    pickled_model = os.path.join(models_cache_dir, 'rfc_model.pkl')
    file_path = os.path.join(source_dir, 'review_400000_samples.json')

    data_loader = DataLoader()
    _data, _labels, _filenames = data_loader.get_data_and_labels(file_path)

    vectorizer = DataVectorizer()

    if retrain_model:
        data = vectorizer.get_vectorized_data(_data)
        labels = _labels

        X_train, X_val, y_train, y_val = train_test_split(data, labels, random_state=0)

        model = RandomForestClassifier()
        print('training model')
        model.fit(X_train, y_train)

        pickle.dump(model, open(pickled_model, 'wb'))

    else:
        print('loading model')
        model = pickle.load(open(pickled_model, 'rb'))

    return model, vectorizer


if __name__ == '__main__':
    initialize_model(retrain_model=True)
