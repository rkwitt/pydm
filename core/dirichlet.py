"""dirichlet.py

This module implements sampling, moment matching and log-likelihood
computation for a Dirichlet distribution with parameter vector alpha.

For details on the Dirichlet distribution, see

http://en.wikipedia.org/wiki/Dirichlet_distribution
"""

# Author: Roland Kwitt, Kitware Inc., 2013
# E-Mail: roland.kwitt@kitware.com


import numpy as np

from scipy.special import (psi, polygamma, gammaln)

def sample(alpha,N=1):
    """ Draw a random sample form a Dirichlet distribution.
    
    Parameters
    ----------
    
    alpha : numpy array, shape (1, D)
            Dirichlet distribution parameters.

    Returns
    -------
    
    X : numpy matrix, shape (N, D)
        Random samples drawn from a Dirichlet distribution. 
    """

    return np.matrix(np.random.dirichlet(alpha,N))
    

def logp(X,alpha,do_sum = False):
    """Compute log-likelihood of sample.
    
    Paramters
    ---------
    
    X : numpy matrix, shape (N, D)
        Data samples.

    alpha : numpy array, shape (1, D)
            Dirichlet distribution parameter vector.
    
    Returns
    -------
    
    L : numpy array, shape (N, 1) 
        Log-Likelihood values of X under the given parameters.
    """
    
    n,d = X.shape
    t0 = np.log(X) * (np.matrix(alpha)-1).transpose()
    t1 = gammaln(np.sum(alpha)) - np.sum(gammaln(alpha))
    if do_sum:
        return np.sum(np.array(t0 + t1))
    return np.array(t0 + t1)


def moment_match(X):
    """Dirichlet moment matching to obtain a rough param. estimate.

    Moment estimation is based on Tom Minka's working paper
    "Estimating a Dirichlet distribution", available online. 

    Parameters
    ----------
    
    X : numpy matrix, shape (N, D) 
        N D-dimensional data samples.

    Returns
    -------
    
    alpha : numpy array, shape (1, D) 
            Moment estimate of the Dirichlet distribution parameter vector.
    """

    # Column-wise mean
    m_0 = np.mean(X,axis=0)
    # Column-wise mean of squared samples
    m_2 = np.mean(np.asarray(X) * np.asarray(X),axis=0)
   
    non_zero = np.where(m_0 > 0)[0]
    t0 = m_0[non_zero] - m_2[non_zero]
    t1 = m_2[non_zero] - m_0[non_zero]**2
    return m_0 *  np.median(t0 / t1)

