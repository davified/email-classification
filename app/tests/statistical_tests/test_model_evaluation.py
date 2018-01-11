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
        self.model, vectorizer = train_yelp.initialize_model(retrain_model=False)

        # Load validation data
        tests_dir = os.path.dirname(os.path.abspath(__file__))
        source_dir = os.path.join(tests_dir, '../../../data/yelp/dataset/review_400000_samples.json')

        data_loader = DataLoader()
        _data, _labels, _filenames = data_loader.get_data_and_labels(source_dir)

        vectorizer = DataVectorizer()
        data = vectorizer.get_vectorized_data(_data)
        labels = _labels

        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(data, labels, random_state=0)

    def test_should_have_recall_score_above_minimum_threshold(self):
        THRESHOLD = 0.85

        y_predicted = self.model.predict(self.X_val)
        recall = metrics.recall_score(self.y_val, y_predicted.round())

        print('recall score: {}'.format(recall))
        self.assertTrue(recall > THRESHOLD, "recall of {} is below threshold of {}".format(recall, THRESHOLD))

    def test_should_have_precision_score_above_minimum_threshold(self):
        THRESHOLD = 0.85

        y_predicted = self.model.predict(self.X_val)
        precision = metrics.precision_score(self.y_val, y_predicted.round())

        print('precision score: {}'.format(precision))
        self.assertTrue(precision > THRESHOLD, "precision of {} is below threshold of {}".format(precision, THRESHOLD))

    # add statistical_test property tag so that we can skip these tests in run_unit_tests.sh
    test_should_have_recall_score_above_minimum_threshold.statistical_test = True
    test_should_have_precision_score_above_minimum_threshold.statistical_test = True
