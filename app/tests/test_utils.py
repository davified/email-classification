import os, unittest
from app.utils import load_files_in_directory


class TestUtils(unittest.TestCase):
    def test_data_loader_should_load_data(self):
        project_dir = os.path.dirname(os.path.abspath(__file__))
        source_dir = os.path.join(project_dir, '../../data/enron_with_categories/1')
        categories = load_files_in_directory(source_dir, '.cats')
        self.assertEqual(len(categories), 834)
        self.assertTrue(isinstance(categories, dict))
        self.assertEqual(categories[10425], '1,1,1\n2,6,1\n2,13,1\n3,3,1\n')
