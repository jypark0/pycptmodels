import numpy as np


class ParametricFlowLine:
    def __init__(self,
                 flow=[
                     [1, 1, 1, 1, 1, 2, 2, 3, 4, 3, 3, 3, 3, 2, 2, 1],
                     [1, 1, 1, 1, 1, 1, 2, 2, 4, 3, 3, 3, 3, 2, 2, 1],
                     [1, 1, 1, 1, 2, 2, 2, 1, 2, 3, 4, 3, 3, 3, 3, 2, 2, 1], ],
                 R=[
                     [1, 2, 2, 2, 2, 1, 2, 2, 1, 2, 2, 2, 3, 2, 2, 1],
                     [1, 2, 2, 2, 2, 2, 2, 1, 1, 2, 2, 2, 3, 2, 2, 1],
                     [1, 2, 2, 2, 1, 2, 2, 2, 2, 2, 1, 2, 2, 2, 3, 2, 2, 1], ],
                 PT=[
                     [0, 80, 90, 60, 65, 50, 90, 60, 100, 90, 60, 90, 130, 90, 60, 0],
                     [0, 80, 90, 60, 65, 90, 60, 50, 100, 90, 60, 90, 130, 90, 60, 0],
                     [0, 80, 90, 60, 50, 90, 60, 65, 90, 60, 100, 90, 60, 90, 130, 90, 60, 0], ],
                 buffer_R=[1, 1, 16],
                 move=3,
                 pick=1):
        """ Create parametric flow line model of CPT

        :param flow: Process flows. Must be a list for each lot class, where each list contains cluster indices
        Positive integer for each cluster with a robot arm, starting from left to right.
        1 for indexer
        :type flow: list of list

        :param R: Redundancies of process flow. Must be a list for each lot class
        :type R: list of list

        :param PT: Process times for process flow
        :type PT: list of list

        :param buffer_R: Buffer sizes, starting from left to right. Length of list must be equal to max cluster index - 1.
        :type buffer_R: list of int

        :param move: Robot move time
        :type move: int

        :param pick: Robot pick time
        :type pick: int
        """
        # K = Number of lot classes
        self.K = len(flow)

        self.flow = flow
        self.R = R
        self.PT = PT
        self.buffer_R = buffer_R
        self.move = move
        self.pick = pick

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

    def run(self, input):
        maxR = np.max(self.R)
        self.X = np.zeros((maxR + np.sum(input.W), len(self.flow[0])))
        self.X[0:maxR, :] = float("-inf")
        self.X = self.X.tolist()

        self.S = [0] * input.N
        self.C = [0] * input.N

        self.CT = [0] * input.N
        self.LRT = [0] * input.N
        self.TT = [0] * input.N

        # Set parameters
        k_w = np.repeat(input.lotclass, input.W)
        # Ensure that k_w[0:maxR] != input.lotclass[0] so that the first lot undergoes prescan setup
        k_w = np.concatenate([np.full(maxR, (input.lotclass[0] + 1) % self.K), k_w])
        self.buffer = [np.where(np.array(self.flow[k]) == -1)[0].tolist() for k in range(self.K)]

        self.last_prescan = [0, 0, 0]
        for k in range(self.K):
            m = self.BN[k] - 1
            while self.flow[k][m] == -1:
                m = m - 1
            self.last_prescan[k] = m

        # Check if prescan setups used
        if any(input.prescan_params):
            prescan = True
        else:
            prescan = False

        wfr = maxR
        # For lot l
        for l in range(input.N):
            # For wafer w in lot l
            for w in range(input.W[l]):
                # For each module
                for m in range(len(self.flow[0])):
                    # Calculate R'(w, m)
                    if k_w[wfr] == k_w[wfr - 1] or m in self.buffer[k_w[wfr]]:
                        R_prime = self.R[k_w[wfr]][m]
                    else:
                        R_prime = 1
                    # Calculate P(w) and tau_s
                    if prescan and k_w[wfr] != k_w[wfr - 1]:
                        P = self.last_prescan[k_w[wfr - 1]]
                        tau_s = input.tau_S[l]
                    else:
                        P = 0
                        tau_s = 0
                    # Calculate tau_r
                    if w == 0 and m == self.BN[k_w[wfr]] + 1:
                        tau_r = input.tau_R[l]
                    else:
                        tau_r = 0

                    # EEEs
                    # First module
                    if m == 0:
                        self.X[wfr][m] = max(input.A[l], self.X[wfr - R_prime][P + 1], self.X[wfr - 1][1]) + tau_s
                    # Last module
                    elif m == len(self.flow[0]) - 1:
                        self.X[wfr][m] = max(self.X[wfr][m - 1] + self.PT[k_w[wfr]][m - 1],
                                             self.X[wfr - R_prime][m] + self.PT[k_w[wfr]][m], self.X[wfr - 1][m])
                    else:
                        self.X[wfr][m] = max(self.X[wfr][m - 1] + self.PT[k_w[wfr]][m - 1] + tau_r,
                                             self.X[wfr - R_prime][m + 1], self.X[wfr - 1][m])
                wfr = wfr + 1

            self.S[l] = self.X[wfr - input.W[l]][self.dummy[input.lotclass[l]]]
            self.C[l] = self.X[wfr - 1][-1] + self.PT[input.lotclass[l]][-1]

            self.CT[l] = self.S[l] - input.A[l]
            self.LRT[l] = self.C[l] - self.S[l]
            self.TT[l] = min(self.C[l] - self.C[l - 1], self.LRT[l]) if l != 0 else self.LRT[l]

        # Delete unneeded X
        del self.X[0:maxR]
