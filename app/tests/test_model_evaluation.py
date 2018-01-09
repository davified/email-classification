import os
import unittest

from sklearn.model_selection import train_test_split

from app.machine_learning_models.CustomRandomForestClassifier import CustomRandomForestClassifier
from app.machine_learning_models.LSTMModel import LSTMModel
from app.pipeline.data_loader import DataLoader
from app.pipeline.data_vectorizer import DataVectorizer


class TestModelEvaluation(unittest.TestCase):
    def setUp(self):
        INPUT_LENGTH=1000
        NO_OF_OUTPUTS=17

        project_dir = os.path.dirname(os.path.abspath(__file__))
        source_dir = os.path.join(project_dir, '../../data/enron_with_categories/1')

        data_loader = DataLoader(source_dir)
        _data, _labels, _filenames = data_loader.get_data_and_labels()

        vectorizer = DataVectorizer(_data, _labels)
        data, labels = vectorizer.get_vectorized_data_and_labels()

        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(data, labels, random_state=0)

        self.model = LSTMModel(input_length=INPUT_LENGTH, no_of_outputs=NO_OF_OUTPUTS)
        self.model.train(self.X_train, self.y_train, epochs=1)

    @unittest.skip
    def test_should_have_recall_score_above_minimum_threshold(self):
        recall = self.model._calculate_recall_score(self.X_val, self.y_val)
        self.assertTrue(recall > 0.85, "recall of {} is below threshold".format(recall))

    @unittest.skip
    def test_should_have_precision_score_above_minimum_threshold(self):
        precision = self.model._calculate_precision_score(self.X_val, self.y_val)
        self.assertTrue(precision > 0.85, "recall of {} is below threshold".format(precision))
