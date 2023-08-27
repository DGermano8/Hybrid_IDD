import unittest
import scipy.stats as stats
import numpy as np


class TestScipyStats(unittest.TestCase):

    def setUp(self):
        self.test_data = np.array([1, 2, 3, 4, 5])

    def test_mean(self):
        self.assertEqual(np.mean(self.test_data), 3.0)

    def test_min_max(self):
        self.assertEqual(np.min(self.test_data), 1)
        self.assertEqual(np.max(self.test_data), 5)

    def test_bogus(self):
        with self.assertRaises(AttributeError):
            self.test_data.fake()


if __name__ == '__main__':
    unittest.main()
