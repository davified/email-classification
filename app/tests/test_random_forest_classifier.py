from unittest import TestCase
from sklearn import ensemble
from app.machine_learning_models.CustomRandomForestClassifier import CustomRandomForestClassifier


class TestRandomForestClassifier(TestCase):
    def setUp(self):
        self.test_model = CustomRandomForestClassifier()

    def test_should_initialize_random_forest_model_upon_instantiation(self):
        self.assertIsNotNone(self.test_model)
        model_name = str(self.test_model.classifier.__class__)
        self.assertTrue(ensemble.RandomForestClassifier.__name__ in model_name)
