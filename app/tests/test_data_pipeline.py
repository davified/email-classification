from unittest import TestCase
from collections import OrderedDict
from app.pipeline.data_pipeline import DataPipeline


class TestDataPipeline(TestCase):
    def setUp(self):
        self.sorted_emails = OrderedDict({'1': 'email sample 1', '2': 'email sample 2', '3': 'email sample 3'})
        self.sorted_categories = OrderedDict({'1': [0, 1, 0], '2': [1, 1, 1], '3': [0, 0, 1]})

    def test_get_data_and_labels(self):
        pipeline = DataPipeline()
        emails_list, categories_list, filenames_list = pipeline.get_data_and_labels(self.sorted_emails,
                                                                                    self.sorted_categories)
        self.assertEqual(len(emails_list), len(categories_list))
        self.assertEqual(len(emails_list), len(filenames_list))
        self.assertEqual(emails_list, ['email sample 1', 'email sample 2', 'email sample 3'])
        self.assertEqual(categories_list, [[0, 1, 0], [1, 1, 1], [0, 0, 1]])
        self.assertEqual(filenames_list, ['1', '2', '3'])
