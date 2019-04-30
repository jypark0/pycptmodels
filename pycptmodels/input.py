import numpy as np
import random

class Input:
  def __init__(self, N, lambda_, lotsizes, lotsize_weights, reticle, prescan, K):
    ''' Create Input object
    
    :param N: Number of lots
    :type N: int

    :param lambda_: Parameter for Poisson process arrival times of lots
    :type lambda: float

    :param lotsizes: Lot sizes
    :type lotsizes: list of int

    :param lotsize_weights: Probabilities of lot sizes corresponding to `lotsizes`
    :type lotsize_weights: list of float

    :param reticle: Parameters for reticle alignment setup distribution
    :type reticle: list of float

    :param prescan: Parameters for prescan setup distribution
    :type prescan: list of float

    :param K: Number of lot classes
    :type K: int

    '''
    self.N = N
    self.lambda_ = lambda_
    self.lotsizes = lotsizes
    self.lotsize_weights = lotsize_weights
    self.reticle_params = reticle
    self.prescan_params = prescan
    self.K = K
  
  def sample_arrivals(self):
    ''' Poisson Process lot arrival times
    '''
    X_t = np.random.exponential(self.lambda_, self.N-1)
    arrivals = np.cumsum(X_t)
    arrivals = np.insert(arrivals, 0, 0.)
    self.A = arrivals.tolist()

  def sample_lotsizes(self):
    ''' Generate lot sizes according to probabilities
    '''
    self.W = random.choices(self.lotsizes, self.lotsize_weights, k = self.N)

  def sample_reticle(self):
    ''' Generate reticle alignment setup times from uniform distribution
    '''
    self.tau_R = np.random.uniform(*self.reticle_params, size=self.N).tolist()

  def sample_prescan(self):
    ''' Generate prescan setup times from uniform distribution
    '''
    self.tau_S = np.random.uniform(*self.prescan_params, size=self.N).tolist()

  def sample_lotclass(self):
    ''' Generate random lot classes
    '''
    self.lotclass = np.random.randint(1, self.K + 1, size=self.N).tolist()

  def Initialize(self):
    self.sample_arrivals()
    self.sample_lotsizes()
    self.sample_lotclass()
    self.sample_reticle()
    self.sample_prescan()


input = Input(N=10, lambda_=2000, lotsizes=[23,24,25], lotsize_weights=[0.25,0.5,0.25], reticle=[210,260], prescan=[0,0], K=3)
input.Initialize()

print("here")