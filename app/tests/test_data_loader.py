import unittest
from collections import OrderedDict
from app.pipeline.data_loader import DataLoader


class TestDataLoader(unittest.TestCase):
    def setUp(self):
        self.sorted_emails = OrderedDict({'1': 'email sample 1', '2': 'email sample 2', '3': 'email sample 3'})
        self.sorted_categories = OrderedDict({'1': [0, 1, 0], '2': [1, 1, 1], '3': [0, 0, 1]})

    def test_get_data_and_labels(self):
        data_loader = DataLoader(self.sorted_emails,
                                 self.sorted_categories)
        emails_list, categories_list, filenames_list = data_loader.get_data_and_labels()
        self.assertEqual(len(emails_list), len(categories_list))
        self.assertEqual(len(emails_list), len(filenames_list))
        self.assertEqual(emails_list, ['email sample 1', 'email sample 2', 'email sample 3'])
        self.assertEqual(categories_list, [[0, 1, 0], [1, 1, 1], [0, 0, 1]])
        self.assertEqual(filenames_list, ['1', '2', '3'])
