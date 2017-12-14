import os
import unittest

from sklearn.model_selection import train_test_split

from app.machine_learning_models.CustomRandomForestClassifier import CustomRandomForestClassifier
from app.pipeline.data_loader import DataLoader
from app.pipeline.data_vectorizer import DataVectorizer


class TestModelEvaluation(unittest.TestCase):
    def test_should_have_recall_score_above_minimum_threshold(self):
        project_dir = os.path.dirname(os.path.abspath(__file__))
        source_dir = os.path.join(project_dir, '../../data/enron_with_categories/1')

        data_loader = DataLoader(source_dir)
        _data, _labels, _filenames = data_loader.get_data_and_labels()

        vectorizer = DataVectorizer(_data, _labels)
        data, labels = vectorizer.get_vectorized_data_and_labels()

        X_train, X_val, y_train, y_val = train_test_split(data, labels, random_state=0)

        rfc_model = CustomRandomForestClassifier()
        rfc_model.train(X_train, y_train)
        recall = rfc_model._calculate_recall_score(X_val, y_val)
        self.assertTrue(recall > 0.85, "recall of {} is below threshold".format(recall))
