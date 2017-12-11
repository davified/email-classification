import os, unittest
from app.utils import load_files_as_dict, sort_dictionary_by_keys, simplify_categories


class TestUtils(unittest.TestCase):
    def setUp(self):
        project_dir = os.path.dirname(os.path.abspath(__file__))
        source_dir = os.path.join(project_dir, '../../data/enron_with_categories/1')
        self.categories = load_files_as_dict(source_dir, '.cats')

    def test_data_loader_should_load_data(self):

        self.assertEqual(len(self.categories), 834)
        self.assertTrue(isinstance(self.categories, dict))
        self.assertEqual(self.categories[10425], '1,1,1\n2,6,1\n2,13,1\n3,3,1\n')

    def test_sort_dictionary_by_keys(self):
        sorted_categories = sort_dictionary_by_keys(self.categories)
        sorted_categories_keys = list(sorted_categories.keys())

        self.assertEqual(len(sorted_categories), len(self.categories))
        self.assertTrue(sorted_categories_keys[1] > sorted_categories_keys[0])
        self.assertTrue(sorted_categories_keys[2] > sorted_categories_keys[1])

    def test_simplify_categories_should_return_dictionary_with_filename_and_list_of_categories(self):
        simple_categories = simplify_categories(self.categories)
        self.assertEqual(len(simple_categories), len(self.categories))
        self.assertTrue(isinstance(simple_categories, dict))
        self.assertTrue(len(list(simple_categories.values())[0]) > 0)

