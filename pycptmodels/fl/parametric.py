import csv

import numpy as np


class ParametricFlowLine:
    def __init__(self, flow=None, R=None, PT=None, move=None, buffer_R=None, pick=None):
        """ Create parametric flow line model of CPT

        :param flow: Process flows. Must be a list for each lot class, where each list contains cluster indices
        Positive integer for each cluster with a robot arm, starting from left to right.
        1 for indexer
        :type flow: list of list

        :param R: Redundancies of process flow. Must be a list for each lot class
        :type R: list of list

        :param PT: Process times for process flow
        :type PT: list of list

        :param buffer_R: Buffer sizes, starting from left to right. Length of list must be equal to max cluster index-1.
        :type buffer_R: list of int

        :param move: Robot move time
        :type move: int

        :param pick: Robot pick time
        :type pick: int
        """

        # Set mutable default arguments
        if flow is None:
            flow = [
                [1, 1, 1, 1, 1, 2, 2, 3, 4, 3, 3, 3, 3, 2, 2, 1],
                [1, 1, 1, 1, 1, 1, 2, 2, 4, 3, 3, 3, 3, 2, 2, 1],
                [1, 1, 1, 1, 2, 2, 2, 1, 2, 3, 4, 3, 3, 3, 3, 2, 2, 1], ]
        if R is None:
            R = [
                [1, 2, 2, 2, 2, 1, 2, 2, 1, 2, 2, 2, 3, 2, 2, 1],
                [1, 2, 2, 2, 2, 2, 2, 1, 1, 2, 2, 2, 3, 2, 2, 1],
                [1, 2, 2, 2, 1, 2, 2, 2, 2, 2, 1, 2, 2, 2, 3, 2, 2, 1], ]
        if PT is None:
            PT = [
                [0, 80, 90, 60, 65, 50, 90, 60, 100, 90, 60, 90, 130, 90, 60, 0],
                [0, 80, 90, 60, 65, 90, 60, 50, 100, 90, 60, 90, 130, 90, 60, 0],
                [0, 80, 90, 60, 50, 90, 60, 65, 90, 60, 100, 90, 60, 90, 130, 90, 60, 0], ]
        if move is None:
            move = 3
        if buffer_R is None:
            buffer_R = [1, 1, 16],
        if pick is None:
            pick = 1

        # K = Number of lot classes
        self.K = len(flow)

        self.flow = flow
        self.R = R
        self.PT = PT
        self.buffer_R = buffer_R
        self.move = move
        self.pick = pick

        self.dummy = []
        self.BN = []
        self.PBN = []
        self.buffer = [[]]
        self.last_prescan = []

        self.X = [[]]
        self.S = []
        self.C = []
        self.L = []
        self.S_w = []
        self.C_w = []

        self.CT = []
        self.LRT = []
        self.TT = []

    def initialize(self):
        """ Create new process flows including buffers for parametric flow line. Modify process times
        """

        # Add buffers to flow, R, and PT
        for k in range(self.K):

            m = 0
            orig_len = len(self.flow[k])
            for input_m in range(orig_len - 1):
                # Check if m < m+1, different clusters prescan case
                # Add as many buffers as the difference in clusters
                for c in range(self.flow[k][m], self.flow[k][m + 1]):
                    # Buffer flow = -1
                    self.flow[k].insert(m + 1, -1)

                    # Get buffer size
                    b = self.buffer_R[c - 1]
                    self.R[k].insert(m + 1, max(b - 1, 1))

                    self.PT[k].insert(m + 1, 0)

                    # Increment to skip just added buffer
                    m = m + 1

                # Check if m > m+1, different clusters postscan case
                # Add as many buffers as the difference in clusters
                for c in range(self.flow[k][m], self.flow[k][m + 1], -1):
                    # Buffer flow = -1
                    self.flow[k].insert(m + 1, -1)

                    # Outlet buffer size = 1
                    self.R[k].insert(m + 1, 1)

                    self.PT[k].insert(m + 1, 0)

                    # Increment to skip just added buffer
                    m = m + 1

                m = m + 1

        # Add dummy modules to match maximum flow length
        max_len = max(len(f) for f in self.flow)
        self.dummy = np.zeros(self.K, dtype=int).tolist()
        for k in range(self.K):
            self.dummy[k] = max_len - len(self.flow[k])
            if self.dummy[k] > 0:
                # Use -2 for dummy modules in flow
                self.flow[k] = ([-2] * self.dummy[k]) + self.flow[k]
                self.R[k] = ([1] * self.dummy[k]) + self.R[k]
                self.PT[k] = ([0] * self.dummy[k]) + self.PT[k]

        # Modify process times
        # Argmax to get bottleneck and penultimate bottleneck modules
        L = np.argsort(-np.divide(self.PT, self.R), axis=1)
        self.BN = L[:, 0].tolist()
        self.PBN = L[:, 1].tolist()
        PT_to_add = np.zeros_like(self.PT[0])
        for k in range(self.K):
            PT_to_add[self.dummy[k]:-1] = self.move + 2 * self.pick
            PT_to_add[self.BN[k]] = 2 * self.move + 4 * self.pick
            PT_to_add[self.PBN[k]] = 3 * self.move + 4 * self.pick
            self.PT[k] = np.add(self.PT[k], PT_to_add).tolist()

        # Initialize other parameters
        self.buffer = [np.where(np.array(self.flow[k]) == -1)[0].tolist() for k in range(self.K)]
        self.last_prescan = [0, 0, 0]
        for k in range(self.K):
            m = self.BN[k] - 1
            while self.flow[k][m] == -1:
                m = m - 1
            self.last_prescan[k] = m

    def run(self, input_sample):
        maxR = np.max(self.R)
        self.X = np.zeros((maxR + np.sum(input_sample.W), len(self.flow[0])))
        self.X[0:maxR, :] = float("-inf")
        self.X = self.X.tolist()

        # S, C, and L of lots
        self.S = [0] * input_sample.N
        self.C = [0] * input_sample.N
        self.L = [0] * input_sample.N

        # S and C of wafers
        self.S_w = np.zeros(np.sum(input_sample.W)).tolist()
        self.C_w = np.zeros(np.sum(input_sample.W)).tolist()

        self.CT = [0] * input_sample.N
        self.LRT = [0] * input_sample.N
        self.TT = [0] * input_sample.N

        # Check if prescan setups used
        if any(input_sample.prescan_params):
            prescan = True
        else:
            prescan = False

        wfr = maxR
        # For lot l
        for lot in range(input_sample.N):
            curr_k = input_sample.lotclass[lot]
            if lot == 0:
                # Ensure that first lot undergoes prescan setup
                prev_k = (input_sample.lotclass[lot] + 1) % self.K
            else:
                prev_k = input_sample.lotclass[lot - 1]

            # For wafer w in lot l
            for w in range(input_sample.W[lot]):
                if w != 0:
                    prev_k = curr_k
                # For each module
                for m in range(len(self.flow[0])):
                    # Calculate R'(w, m)
                    if curr_k == prev_k or m in self.buffer[curr_k]:
                        R_prime = self.R[curr_k][m]
                    else:
                        R_prime = 1
                    # Calculate P(w) and tau_s
                    if prescan and curr_k != prev_k:
                        P = self.last_prescan[prev_k]
                        tau_s = input_sample.tau_S[lot]
                    else:
                        P = 0
                        tau_s = 0
                    # Calculate tau_r
                    if w == 0 and m == self.BN[curr_k] + 1:
                        tau_r = input_sample.tau_R[lot]
                    else:
                        tau_r = 0

                    # EEEs
                    # First module
                    if m == 0:
                        self.X[wfr][m] = max(input_sample.A[lot], self.X[wfr - R_prime][P + 1],
                                             self.X[wfr - 1][1]) + tau_s
                    # Last module
                    elif m == len(self.flow[0]) - 1:
                        self.X[wfr][m] = max(self.X[wfr][m - 1] + self.PT[curr_k][m - 1],
                                             self.X[wfr - R_prime][m] + self.PT[curr_k][m], self.X[wfr - 1][m])
                    else:
                        self.X[wfr][m] = max(self.X[wfr][m - 1] + self.PT[curr_k][m - 1] + tau_r,
                                             self.X[wfr - R_prime][m + 1], self.X[wfr - 1][m])

                self.S_w[wfr - maxR] = self.X[wfr][self.dummy[curr_k]]
                self.C_w[wfr - maxR] = self.X[wfr][-1] + self.PT[curr_k][-1]
                wfr = wfr + 1

            # Calculate lot loading times
            prev_k = input_sample.lotclass[lot - 1]
            if lot == 0:
                self.L[lot] = 0.
            else:
                if curr_k == prev_k:
                    self.L[lot] = self.X[wfr - input_sample.W[lot] - self.R[prev_k][0]][1]
                else:
                    self.L[lot] = self.X[wfr - input_sample.W[lot] - 1][1]
            self.S[lot] = self.X[wfr - input_sample.W[lot]][self.dummy[curr_k]]
            self.C[lot] = self.X[wfr - 1][-1] + self.PT[curr_k][-1]

            self.CT[lot] = self.C[lot] - input_sample.A[lot]
            self.LRT[lot] = self.C[lot] - self.S[lot]
            self.TT[lot] = min(self.C[lot] - self.C[lot - 1], self.LRT[lot]) if lot != 0 else self.LRT[lot]

        # Delete unneeded X
        del self.X[0:maxR]

    def csv_write_lot(self, filename):
        with open(filename, 'w', encoding='utf-8') as f:
            writer = csv.writer(f)
            for l, s, c, in zip(self.L, self.S, self.C):
                writer.writerow((l, s, c))

    def csv_write_wfr(self, filename):
        with open(filename, 'w', encoding='utf-8') as f:
            writer = csv.writer(f)
            for s, c, x, in zip(self.S_w, self.C_w, self.X):
                writer.writerow((s, c, *x))
