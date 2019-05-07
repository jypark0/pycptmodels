import numpy as np
import random
import csv
import math


class PoissonProcessInput:
    def __init__(self, N, lambda_, lotsizes, lotsize_weights, reticle, prescan, K):
        """ Create Input object

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

        """
        self.N = N
        self.lambda_ = lambda_
        self.lotsizes = lotsizes
        self.lotsize_weights = lotsize_weights
        self.reticle_params = reticle
        self.prescan_params = prescan
        self.K = K

    def _sample_arrivals(self):
        """ Poisson Process lot arrival times
        """
        X_t = np.random.exponential(self.lambda_, self.N - 1)
        arrivals = np.cumsum(X_t)
        arrivals = np.insert(arrivals, 0, 0.)
        # self.A = arrivals.tolist()
        self.A = [int(round(x, -2)) for x in arrivals.tolist()]


    def _sample_lotsizes(self):
        """ Generate lot sizes according to probabilities
        """
        self.W = random.choices(self.lotsizes, self.lotsize_weights, k=self.N)

        # Calculate first wafer index (omega(l, 1)) for all lots
        self.first_wfr_idx = np.subtract(np.cumsum(self.W), self.W)

    def _sample_reticle(self):
        """ Generate reticle alignment setup times from uniform distribution
        """
        self.tau_R = np.random.uniform(*self.reticle_params, size=self.N).tolist()

    def _sample_prescan(self):
        """ Generate prescan setup times from uniform distribution
        """
        self.tau_S = np.random.uniform(*self.prescan_params, size=self.N).tolist()

    def _sample_lotclass(self):
        """ Generate random lot classes
        """
        self.lotclass = np.random.randint(0, self.K, size=self.N).tolist()

    def initialize(self):
        self._sample_arrivals()
        self._sample_lotsizes()
        self._sample_lotclass()
        self._sample_reticle()
        self._sample_prescan()

    def write_csv(self, filename):
        with open(filename, 'w', encoding='utf-8') as f:
            writer = csv.writer(f)
            for a, w, c, r, s in zip(self.A, self.W, self.lotclass, self.tau_R, self.tau_S):
                writer.writerow((a, w, c, r, s))
