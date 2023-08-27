import unittest
import scipy.stats as stats
import numpy as np


class TestScipyLognormal(unittest.TestCase):
    """
    The scipy.stats.lognorm API requires specifying mu and sigma in a
    strange way.
    """

    def setUp(self):
        self.pdf = lambda x, mu, sig: np.exp(- (np.log(x) - mu)**2 / (2 * sig**2)) / (x * sig * np.sqrt(2 * np.pi))
        self.x = np.array([1E-6, 1.0, 2.0, 5.0, 6.789])
        self.th1 = {'mu': 1.0, 'sigma': 1.0}
        self.th2 = {'mu': 2.0, 'sigma': 2.3}
        self.parameter_sets = [self.th1, self.th2]

    def test_pdf(self):
        for th in self.parameter_sets:
            scipy_vals = stats.lognorm.pdf(self.x, th['sigma'], scale = np.exp(th['mu']))
            my_vals = self.pdf(self.x, th['mu'], th['sigma'])

            for ix in range(len(self.x)):
                self.assertAlmostEqual(scipy_vals[ix], my_vals[ix])


if __name__ == '__main__':
    unittest.main()
