"""dirichlet_test.py

Testing for the Dirichlet distribution module.
"""


# Author: Roland Kwitt, Kitware Inc., 2013
# E-Mail: roland.kwitt@kitware.com


import os
import sys
import unittest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import core.dirichlet as dirichlet


class DirichletTesting(unittest.TestCase):
    """Testcase for the dirichlet distribution module.
    """

    def setUp(self):
        """Load data sample.
        """
        self._data = np.genfromtxt("data.sample")
    
    
    def test_moment_match(self):
        """Test the moment matching code.
        """
        a_hat = dirichlet.moment_match(self._data) 
        self.assertTrue(np.allclose(a_hat,np.array([1.30404473,1.21440784,9.86841359])))


    def test_logp(self):
        """Test log-likelihood computation.
        """
        ll = dirichlet.logp(self._data,np.array([1,1,8]),do_sum=True)
        self.assertTrue(np.allclose(ll,261.218091569))


if __name__ == '__main__':
    unittest.main()
