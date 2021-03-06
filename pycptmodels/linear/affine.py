import csv

import numpy as np


class AffineModel:
    def __init__(self):
        """Create affine model. Instance variables are initially empty lists.
        """
        self.A = []
        self.B = []

        self.S = []
        self.C = []

        self.CT = []
        self.LRT = []
        self.TT = []

    def train(self, input_sample, C_l, C_w):
        """Train using an Input object and other data obtained from another model.
        Calculates parameters A and B.

        :param input_sample: input to train the model on.
        :type input_sample: pycptmodels.input.Input

        :param C_l: Lot completion times
        :type C_l: list of float

        :param C_w: Wafer completion times
        :type C_w: list of float

        :return: None
        """
        self.A = np.zeros(input_sample.K).tolist()
        A_sum = np.zeros(input_sample.K).tolist()
        A_count = np.zeros(input_sample.K, dtype=int).tolist()

        self.B = np.zeros((input_sample.K, input_sample.K)).tolist()
        B_sum = np.zeros((input_sample.K, input_sample.K)).tolist()
        B_count = np.zeros((input_sample.K, input_sample.K), dtype=int).tolist()

        # Discard first lot
        for lot in range(1, input_sample.N):
            curr_k = input_sample.lotclass[lot]
            prev_k = input_sample.lotclass[lot - 1]

            A_sum[curr_k] += (C_l[lot] - C_w[input_sample.first_wfr_idx[lot]])
            A_count[curr_k] += (input_sample.W[lot] - 1)

            B_sum[curr_k][prev_k] += (C_w[input_sample.first_wfr_idx[lot]] - max(input_sample.A[lot], C_l[lot - 1]))
            B_count[curr_k][prev_k] += 1

        # Average parameters
        for k1 in range(input_sample.K):
            self.A[k1] = A_sum[k1] / A_count[k1] if A_count[k1] else 0.
            for k2 in range(input_sample.K):
                self.B[k1][k2] = B_sum[k1][k2] / B_count[k1][k2] if B_count[k1][k2] else 0.

    def run(self, input_sample):
        """Estimate lot start and completion times of an Input sample. Model must be trained before use.
        Also calculates cycle time, lot residency time, and throughput time of lots.

        :param input_sample: input to simulate the model on.
        :type input_sample: pycptmodels.input.Input

        :return: None
        """
        self.S = np.zeros(input_sample.N).tolist()
        self.C = np.zeros(input_sample.N).tolist()

        self.CT = np.zeros(input_sample.N).tolist()
        self.LRT = np.zeros(input_sample.N).tolist()
        self.TT = np.zeros(input_sample.N).tolist()

        for lot in range(input_sample.N):
            curr_k = input_sample.lotclass[lot]

            if lot == 0:
                # Initialize lot class to be different than lot class of first lot (to ensure prescan setup occurs)
                prev_k = (input_sample.lotclass[lot] + 1) % input_sample.K
                self.S[lot] = input_sample.A[lot]
            else:
                prev_k = input_sample.lotclass[lot - 1]
                self.S[lot] = max(input_sample.A[lot], self.C[lot - 1])

            self.C[lot] = self.S[lot] + self.A[curr_k] * (input_sample.W[lot] - 1) + self.B[curr_k][prev_k]

            # Calculate CT, LRT, TT
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
            writer.writerow(('Lot class', 'A', 'B'))
            for k, a, b in zip(range(len(self.A)), self.A, self.B):
                writer.writerow((k, a, b))

    def csv_write_run(self, filename):
        """Write estimated values to csv file. Train and run model first. Generally used for debugging code.

        :param filename: filename of csv file
        :type filename: str

        :return: None
        """
        with open(filename, 'w', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(('Lot', 'S', 'C', 'CT', 'LRT', 'TT'))
            for lot, s, c, ct, lrt, tt in zip(range(len(self.S)), self.S, self.C, self.CT, self.LRT, self.TT):
                writer.writerow((lot, s, c, ct, lrt, tt))
