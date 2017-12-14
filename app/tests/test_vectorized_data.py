import unittest
import numpy as np
from numpy.testing import assert_array_equal

from app.VectorizedData.vectorized_data import VectorizedData


class TestVectorizedData(unittest.TestCase):
    def setUp(self):
        self.X = np.array([[1, 0, 1, 0],
             [1, 0, 0, 0],
             [0, 0, 0, 0]])
        self.y = np.array([[1, 0], [1, 0], [1, 0]])
        self.data = VectorizedData(self.X, self.y)

    def test_train_test_split_should_return_tuple_of_train_and_test_data(self):
        self.assertEquals(self.data.X_train.shape, (2,4))
        self.assertEquals(self.data.X_val.shape, (1,4))
        self.assertEquals(self.data.y_train.shape, (2,2))
        self.assertEquals(self.data.y_val.shape, (1,2))

    def test_vectorized_data_should_show_summary_of_y_labels(self):
        summary = self.data.get_count(self.y)
        assert_array_equal(summary, [3, 0])
