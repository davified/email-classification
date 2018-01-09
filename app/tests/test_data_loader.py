import os
import unittest
from collections import OrderedDict
from app.pipeline.data_loader import DataLoader


class TestDataLoader(unittest.TestCase):
    def setUp(self):
        project_dir = os.path.dirname(os.path.abspath(__file__))
        source_dir = os.path.join(project_dir, '../../data/enron_with_categories/1')
        self.data_loader = DataLoader(source_dir)

        self.categories = self.data_loader._load_files_as_dict(source_dir, '.cats')
        self.sorted_emails = OrderedDict({'1': 'email sample 1', '2': 'email sample 2', '3': 'email sample 3'})
        self.sorted_categories = OrderedDict({'1': [0, 1, 0], '2': [1, 1, 1], '3': [0, 0, 1]})

    def test_get_data_and_labels(self):

        emails_list, categories_list, filenames_list = self.data_loader.get_data_and_labels()
        self.assertEqual(len(emails_list), len(categories_list))
        self.assertEqual(len(emails_list), len(filenames_list))

    def test_data_loader_should_load_data(self):
        self.assertEqual(len(self.categories), 834)
        self.assertTrue(isinstance(self.categories, dict))
        self.assertEqual(self.categories[10425], '1,1,1\n2,6,1\n2,13,1\n3,3,1\n')

    def test_sort_dictionary_by_keys(self):
        sorted_categories = self.data_loader._sort_dictionary_by_keys(self.categories)
        sorted_categories_keys = list(sorted_categories.keys())

        self.assertEqual(len(sorted_categories), len(self.categories))
        self.assertTrue(sorted_categories_keys[1] > sorted_categories_keys[0])
        self.assertTrue(sorted_categories_keys[2] > sorted_categories_keys[1])

    def test_simplify_categories_should_return_dictionary_with_filename_and_list_of_categories(self):
        simple_categories = self.data_loader._simplify_categories(self.categories)
        self.assertEqual(len(simple_categories), len(self.categories))
        self.assertTrue(isinstance(simple_categories, dict))
        self.assertTrue(len(list(simple_categories.values())) > 0)