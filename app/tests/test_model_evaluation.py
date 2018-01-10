import os
import unittest

from app.pipeline.enron.data_vectorizer import DataVectorizer
from sklearn.model_selection import train_test_split

from app.constants import INPUT_LENGTH, NO_OF_OUTPUTS
from app.machine_learning_models.LSTMModel import LSTMModel
from app.pipeline.enron.data_loader import DataLoader

from app import train_yelp


class TestModelEvaluation(unittest.TestCase):
    def setUp(self):
        # load trained model
        self.model, vectorizer = train_yelp.initialize_model()

        # Load validation data
        project_dir = os.path.dirname(os.path.abspath(__file__))
        source_dir = os.path.join(project_dir, '../../../data/yelp/dataset/review_400000_samples.json')

        data_loader = DataLoader(source_dir)
        _data, _labels, _filenames = data_loader.get_data_and_labels()

        vectorizer = DataVectorizer(_data, _labels)
        data = vectorizer.get_vectorized_data(_data)
        labels = vectorizer.get_vectorized_labels()

        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(data, labels, random_state=0)

    # @unittest.skip
    def test_should_have_recall_score_above_minimum_threshold(self):
        recall = self.model._calculate_recall_score(self.X_val, self.y_val)
        self.assertTrue(recall > 0.85, "recall of {} is below threshold".format(recall))

    # @unittest.skip
    def test_should_have_precision_score_above_minimum_threshold(self):
        precision = self.model._calculate_precision_score(self.X_val, self.y_val)
        self.assertTrue(precision > 0.85, "recall of {} is below threshold".format(precision))

    test_should_have_recall_score_above_minimum_threshold.statistical_test = True
    test_should_have_precision_score_above_minimum_threshold.statistical_test = True
