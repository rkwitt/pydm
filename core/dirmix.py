"""dirmix.py

This module implements a Dirichlet mixture model, as it is used in: 

N. Rasiwasia and N. Vasconcelos, "Holistic Context Models for Visual 
Recognition". In: IEEE Transactions on Pattern Analysis and Machine 
Intelligence, May 2012, 34(5): 902-917
"""


# Author: Roland Kwitt, Kitware Inc., 2013
# E-Mail: roland.kwitt@kitware.com


import numpy as np
import dirichlet
from scipy.special import (psi, digamma, polygamma, gammaln)
from sklearn.cluster import KMeans
from collections import defaultdict


def sample(weights,alphas,N=1):
    """ Draw N random samples from model.
        
    Parameters
    ----------

    weights : numpy array, shape (1, C) 
        Weights for C mixture components (must sum to 1).

    alphas  : numpy matrix, shape (C, D) 
        C D-dimensional Dirichlet distribution parameter vectors.

    N : int
        Number of samples to draw from the Dirichlet mixture model.   
    
    Returns:
    --------
    
    X : numpy matrix, shape (N, D)
        N D-dim. vectors, drawn from the Dirichlet mixture model.

    s : numpy array, shape (N, 1)
        The i-th entry of s identifies which mixture component was
        responsible for generating the sample at X[i,:].
    """
    
    if not np.allclose(np.sum(weights),1.0):
        raise Exception("Weights do not sum up to 1.")

    c,d = alphas.shape
    if not c == len(weights):
        raise Exception("Weights and alphas not compatible")

    cweights = np.cumsum(weights)
    r = np.random.uniform(0,1,N) 
    s = np.zeros([N,1],dtype=np.int)
    X = np.zeros([N,d])

    for i,p in enumerate(r):
        s[i] = np.argmax((p < cweights) == True)
        X[i,:] = np.random.dirichlet(np.asarray(alphas[s[i],:])[0])
    return (np.matrix(X),s)


def init(X,n_components=1):
    """Initialize mixture parameters from a data sample.

    The strategy is to run simple k-means clustering on the
    simplex, get the cluster assignments and then run an in-
    dependent Dirichlet (moment-matching) parameter estimate 
    on the samples belonging to each cluster.

    Parameters
    ----------
    
    X : numpy matrix, shape (N, D) 
        N D-dimensional data samples.
    
    n_components : int
        Number of desired mixture components.

    Returns
    -------
    
    weights : numpy array, shape (n_component, 1)
        Initial weights for each mixture component.
    
    alphas  : numpy matrix, shape (C, D) 
        Matrix where the i-th row contains the D-dimensional Dirichlet
        distribution parameter vector for the i-th mixture component.
    """
    
    n,d = X.shape

    km_obj = KMeans(init='k-means++', n_clusters=n_components, n_init=50)
    km_lab = km_obj.fit_predict(X)
  
    hist = defaultdict(int)
    for x in km_lab: 
        hist[x] += 1

    weights = np.array([hist[key] for key in hist])/float(n)

    alphas = np.zeros([n_components,d])
    for i in range(0,n_components):
        pos = np.array(np.where(km_lab == i))
        alphas[i,:] = dirichlet.moment_match(X[pos.ravel(),:])
    return (weights,alphas)


def estimate(X,n_components=1,n_iter=1,weights_0=None,alphas_0=None):
    """Run GEM algorithm to estimate mixture parameters.

    Parameters
    ----------
    
    X : numpy matrix, shape (N, D)
        N D-dimensional data samples (rows must sum to 1).
    
    n_components : int
        Number of desired mixture components.
    
    n_iter : int
        Maximum number of iterations to run.
    
    weights_0 : numpy array, shape (1, C), optional
        Vector of initial mixture weights (must sum to 1).

    alphas_0 : numpy matrix, shape (n_components, D), optional
        Matrix of initial Dirichlet mixture distribution parameters. If
        provided, the i-th row needs to contain the initial D-dimensional 
        Dirichlet distribution parameter vecotor of the i-th mixture 
        component.
        
    Returns
    -------
    
    weights : numpy array, shape (1, C)
        GEM estimated mixture component weights.
    
    alphas : numpy matrix, shape (C, D)
        GEM estimated parameters of each of the C Dirichlet mixture 
        distributions. The i-th row contains the D-dimensional parameter
        vector of the i-th Dirichlet mixture component.
    """
    
    if weights_0 is None or alphas_0 is None:
        (weights_0, alphas_0) = init(X,n_components)

    n,d0 = X.shape
    c,d1 = alphas_0.shape
    assert c == n_components and d0 == d1, "Oops ..."
    d = d0

    a_hat = alphas_0
    w_hat = weights_0

    for i in range(0,n_iter):
        #-------
        # E-Step
        #-------
        ll_k = np.zeros([n,c])
        for k in range(0,c):
            ll_k[:,k] = (dirichlet.logp(X,a_hat[k,:]) + 
                         np.log(w_hat[k])).ravel()
  
        t0 = np.matrix(np.max(ll_k,axis=1)).transpose()
        t1 = np.exp(ll_k - np.tile(t0,(1,c)))
        t2 = np.sum(t1,axis=1)
        t3 = np.log(t2) + t0
    
        print "[Iteration %.3d]: Log-likelihood=%.5f" % (i,np.sum(t3))

        Y = np.exp(ll_k - np.tile(t3,(1,c)))

        #-------
        # M-Step
        #-------
        N = np.sum(Y)
        N_k = np.asarray(np.sum(Y,axis=0)).ravel()

        w_new = N_k / N   
        a_new = np.zeros([c,d])
        H = np.ones([d,d,c])
        g = np.zeros([c,d])
    
        for k in range(0,c):
            t0 = np.sum(a_hat[k,:])
            H[:,:,k] = np.ones([d,d]) * trigamma(t0) * N_k[k]

            for l in range(0,d):
                H[l,l,k] = N_k[k] * (trigamma(t0) - trigamma(a_hat[k,l]))
                x_l = np.array(np.log(X[:,l])).ravel()
                y_l = np.asarray(Y[:,k]).ravel()
                t1 = np.sum(x_l * y_l)
                g[k,l] = N_k[k] * (digamma(t0) - digamma(a_hat[k,l])) + t1
            
            a_new[k,:] = a_hat[k,:] - np.linalg.solve(H[:,:,k],g[k,:])
        
        a_hat = a_new
        w_hat = w_new
    return (w_hat, a_hat)


def trigamma(x):
    """ Trigramma function.

    For details see: http://en.wikipedia.org/wiki/Trigamma_function
    
    Parameters
    ----------
    
    x : float
        Evaluate Trigamma function at x.

    Returns
    -------

    y : float
        Value of the Trigamma function at x.
    """

    return polygamma(1,x)
    

