import csv

import numpy as np
from sklearn.linear_model import LinearRegression


class LotERM:
    def __init__(self):
        """Create lot-level log exit recursion model (ERM). Instance variables are initially empty lists.
        """
        self.phi1 = []
        self.phi2 = []

        self.A1 = []
        self.B1 = []
        self.A2 = []
        self.B2 = []
        self.Dm = []
        self.Dp = []
        self.E = []

        self.Vm = []
        self.Vp = []
        self.V = []
        self.L = []
        self.S = []
        self.C = []

        self.CT = []
        self.LRT = []
        self.TT = []

    def train(self, input_sample, L_l, S_l, C_l, R):
        """Train using an Input object and other data obtained from another model. The input must contain several
        distinct lot sizes and N must be sufficiently large to discern bottleneck contention from no bottleneck
        contention cases.
        Calculates parameters A1, B1, A2, B2, Dm, Dp, and E.

        :param input_sample: input to train the model on.
        :type input_sample: pycptmodels.input.Input

        :param L_l: Lot load times
        :type L_l: list of float

        :param S_l: Lot start times
        :type S_l: list of float

        :param C_l: Lot completion times
        :type C_l: list of float

        :param R: Redundancies of process flow. Must be a list for each lot class
        :type R: list of int

        :return: None
        """
        phi = np.zeros(input_sample.N, dtype=int)

        # No bottleneck contention
        self.A1 = np.zeros(input_sample.K).tolist()
        self.B1 = np.zeros(input_sample.K).tolist()
        TT1_sum = np.zeros((input_sample.K, len(input_sample.lotsizes))).tolist()
        TT1_count = np.zeros((input_sample.K, len(input_sample.lotsizes)), dtype=int)

        self.A2 = np.zeros(input_sample.K).tolist()
        self.B2 = np.zeros(input_sample.K).tolist()
        TT2_sum = np.zeros((input_sample.K, len(input_sample.lotsizes))).tolist()
        TT2_count = np.zeros((input_sample.K, len(input_sample.lotsizes)), dtype=int)

        self.Dm = np.zeros(input_sample.K).tolist()
        Dm_sum = np.zeros(input_sample.K).tolist()
        Dm_count = np.zeros(input_sample.K, dtype=int).tolist()
        self.Dp = np.zeros(input_sample.K).tolist()

        self.E = np.zeros((input_sample.K, input_sample.K)).tolist()
        E_sum = np.zeros((input_sample.K, input_sample.K)).tolist()
        E_count = np.zeros((input_sample.K, input_sample.K), dtype=int).tolist()

        # Discard first and last lots
        for lot in range(1, input_sample.N - 1):
            curr_k = input_sample.lotclass[lot]
            prev_k = input_sample.lotclass[lot - 1]

            lotsize_idx = input_sample.lotsizes.index(input_sample.W[lot])

            # No bottleneck contention
            if S_l[lot] > C_l[lot - 1]:
                phi[lot] = 1

                TT1_sum[curr_k][lotsize_idx] += (C_l[lot] - S_l[lot])
                TT1_count[curr_k, lotsize_idx] += 1

            # Bottleneck contention
            elif curr_k == prev_k and input_sample.A[lot] <= S_l[lot - 1]:
                phi[lot] = 2

                TT2_sum[curr_k][lotsize_idx] += (C_l[lot] - C_l[lot - 1])
                TT2_count[curr_k, lotsize_idx] += 1

            # E
            E_sum[curr_k][prev_k] += (S_l[lot] - L_l[lot])
            E_count[curr_k][prev_k] += 1

        # Calculate A1, B1, A2, B2 with weighted least squares regression
        mean_TT1 = np.divide(TT1_sum, TT1_count, out=np.zeros_like(TT1_sum), where=TT1_count != 0)
        mean_TT2 = np.divide(TT2_sum, TT2_count, out=np.zeros_like(TT2_sum), where=TT2_count != 0)
        X = np.array(input_sample.lotsizes).reshape((-1, 1))
        for k in range(input_sample.K):
            reg1 = LinearRegression()
            reg2 = LinearRegression()

            nonzero_idx1 = np.nonzero(TT1_sum[k])
            nonzero_idx2 = np.nonzero(TT2_sum[k])

            reg1.fit(X[nonzero_idx1], mean_TT1[k][nonzero_idx1], sample_weight=TT1_count[k][nonzero_idx1])
            reg2.fit(X[nonzero_idx2], mean_TT2[k][nonzero_idx2], sample_weight=TT2_count[k][nonzero_idx2])

            self.A1[k] = reg1.coef_[0]
            self.B1[k] = reg1.predict([[1]])[0]
            self.A2[k] = reg2.coef_[0]
            self.B2[k] = reg2.predict([[1]])[0]

        # Check if last lot phi1 or phi2
        if S_l[-1] > C_l[-2]:
            phi[-1] = 1
        elif input_sample.lotclass[-1] == input_sample.lotclass[-2] and input_sample.A[-1] <= S_l[-2]:
            phi[-1] = 2

        # Calculate vacation time related parameters
        for lot in range(1, input_sample.N - 1):
            curr_k = input_sample.lotclass[lot]

            if phi[lot + 1] == 2:
                Dm_sum[curr_k] += (C_l[lot] - L_l[lot + 1])
                Dm_count[curr_k] += 1

        # Store phi1, phi2 for reference
        self.phi1 = np.where(phi == 1)[0].tolist()
        self.phi2 = np.where(phi == 2)[0].tolist()

        # Average parameters
        for k1 in range(input_sample.K):
            self.Dm[k1] = Dm_sum[k1] / Dm_count[k1] if Dm_count[k1] else 0.
            self.Dp[k1] = self.Dm[k1] - (R[k1][0] - 1) * self.A2[k1]

            for k2 in range(input_sample.K):
                self.E[k1][k2] = E_sum[k1][k2] / E_count[k1][k2] if E_count[k1][k2] else 0.

    def run(self, input_sample):
        """Estimate lot vacation, load, start, and completion times of an Input sample. Model must be trained before
        use. Also calculates cycle time, lot residency time, and throughput time of lots.

        :param input_sample: input to simulate the model on.
        :type input_sample: pycptmodels.input.Input

        :return: None
        """
        self.Vm = np.zeros(input_sample.N).tolist()
        self.Vp = np.zeros(input_sample.N).tolist()
        self.V = np.zeros(input_sample.N).tolist()

        self.L = np.zeros(input_sample.N).tolist()
        self.S = np.zeros(input_sample.N).tolist()
        self.C = np.zeros(input_sample.N).tolist()

        self.CT = np.zeros(input_sample.N).tolist()
        self.LRT = np.zeros(input_sample.N).tolist()
        self.TT = np.zeros(input_sample.N).tolist()

        for lot in range(input_sample.N):
            curr_k = input_sample.lotclass[lot]

            # Lot loading times
            if lot == 0:
                self.L[lot] = input_sample.A[lot]
                self.S[lot] = self.L[lot]
                self.C[lot] = self.S[lot] + self.B1[curr_k] + self.A1[curr_k] * (input_sample.W[lot] - 1)
            else:
                prev_k = input_sample.lotclass[lot - 1]
                if curr_k == prev_k:
                    self.V[lot - 1] = self.Vm[lot - 1]
                else:
                    self.V[lot - 1] = self.Vp[lot - 1]
                self.L[lot] = max(input_sample.A[lot], self.V[lot - 1])
                self.S[lot] = self.L[lot] + self.E[curr_k][prev_k]
                self.C[lot] = max(self.S[lot] + self.B1[curr_k] + self.A1[curr_k] * (input_sample.W[lot] - 1),
                                  self.C[lot - 1] + self.B2[curr_k] + self.A2[curr_k] * (input_sample.W[lot] - 1))

            self.Vm[lot] = self.C[lot] - self.Dm[curr_k]
            self.Vp[lot] = self.C[lot] - self.Dp[curr_k]

            self.CT[lot] = self.C[lot] - input_sample.A[lot]
            self.LRT[lot] = self.C[lot] - self.S[lot]
            self.TT[lot] = min(self.C[lot] - self.C[lot - 1], self.LRT[lot]) if lot != 0 else self.LRT[lot]

    def csv_write_params(self, filename):
        """Write trained parameters to csv file. Train model first. Generally used for debugging code.

        :param filename: filename of csv file
        :type filename: str

        :return: None
        """
        with open(filename, 'w', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(('Lot class', 'A1', 'B1', 'A2', 'B2', 'Dm', 'Dp', 'E'))
            for k, a1, b1, a2, b2, dm, dp, e, in zip(range(len(self.A1)), self.A1, self.B1, self.A2, self.B2, self.Dm,
                                                     self.Dp, self.E):
                writer.writerow((k, a1, b1, a2, b2, dm, dp, e))

    def csv_write_run(self, filename):
        """Write estimated values to csv file. Train and run model first. Generally used for debugging code.

        :param filename: filename of csv file
        :type filename: str

        :return: None
        """
        with open(filename, 'w', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(('Lot', 'V', 'L', 'S', 'C', 'Vm', 'Vp', 'CT', 'LRT', 'TT'))
            for lot, v, l, s, c, vm, vp, ct, lrt, tt in zip(range(len(self.S)), self.V, self.L, self.S, self.C, self.Vm,
                                                            self.Vp, self.CT, self.LRT, self.TT):
                writer.writerow((lot, v, l, s, c, vm, vp, ct, lrt, tt))
