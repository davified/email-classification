import pandas as pd
import unittest
from parser.text_to_csv_parser import TextToCSVParser

class TestTextToCSVParser(unittest.TestCase):
    def setUp(self):
        df = pd.DataFrame(data={'col1': [1, 2, 3], 'col2': [3, 4, 3]})
        self.parser = TextToCSVParser(df)

    def test_simple_demo(self):
        self.assertEqual((3,2), self.parser.df.shape)
