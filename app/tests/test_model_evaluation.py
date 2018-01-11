import os
import unittest
from sklearn import metrics

from app.pipeline.yelp.data_vectorizer import DataVectorizer
from sklearn.model_selection import train_test_split

from app.pipeline.yelp.data_loader import DataLoader

from app import train_yelp


class TestModelEvaluation(unittest.TestCase):
    def setUp(self):
        # load trained model
        self.model, vectorizer = train_yelp.initialize_model(retrain_model=True)

        # Load validation data
        tests_dir = os.path.dirname(os.path.abspath(__file__))
        source_dir = os.path.join(tests_dir, '../../data/yelp/dataset/review_100_samples.json')

        data_loader = DataLoader()
        _data, _labels, _filenames = data_loader.get_data_and_labels(source_dir)

        vectorizer = DataVectorizer(_data, _labels)
        data = vectorizer.get_vectorized_data(_data)
        labels = _labels

        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(data, labels, random_state=0)

    def test_should_have_recall_score_above_minimum_threshold(self):
        THRESHOLD = 0.001

        y_predicted = self.model.predict(self.X_val).argmax(axis=1)
        recall = metrics.recall_score(self.y_val, y_predicted.round())

        self.assertTrue(recall > THRESHOLD, "recall of {} is below threshold of {}".format(recall, THRESHOLD))

    def test_should_have_precision_score_above_minimum_threshold(self):
        THRESHOLD = 0.001

        y_predicted = self.model.predict(self.X_val).argmax(axis=1)
        precision = metrics.precision_score(self.y_val, y_predicted.round())

        self.assertTrue(precision > THRESHOLD, "precision of {} is below threshold of {}".format(precision, THRESHOLD))

    # add statistical_test property tag to be used in run_*_tests.sh scripts
    test_should_have_recall_score_above_minimum_threshold.statistical_test = True
    test_should_have_precision_score_above_minimum_threshold.statistical_test = True
