import csv

import numpy as np


class LinearModel:
    def __init__(self):
        self.A = []

        self.S = []
        self.C = []

        self.CT = []
        self.LRT = []
        self.TT = []

    def train(self, input_sample, C_l):
        self.A = np.zeros(input_sample.K).tolist()
        A_sum = np.zeros(input_sample.K).tolist()
        A_count = np.zeros(input_sample.K, dtype=int).tolist()

        # Discard first lot
        for lot in range(1, input_sample.N):
            curr_k = input_sample.lotclass[lot]

            A_sum[curr_k] += (C_l[lot] - max(input_sample.A[lot], C_l[lot - 1]))
            A_count[curr_k] += (input_sample.W[lot])

        # Average parameters
        for k1 in range(input_sample.K):
            self.A[k1] = A_sum[k1] / A_count[k1] if A_count[k1] else 0.

    def run(self, input_sample):
        self.S = np.zeros(input_sample.N).tolist()
        self.C = np.zeros(input_sample.N).tolist()

        self.CT = np.zeros(input_sample.N).tolist()
        self.LRT = np.zeros(input_sample.N).tolist()
        self.TT = np.zeros(input_sample.N).tolist()

        for lot in range(input_sample.N):
            curr_k = input_sample.lotclass[lot]

            if lot == 0:
                self.S[lot] = input_sample.A[lot]
            else:
                self.S[lot] = max(input_sample.A[lot], self.C[lot - 1])

            self.C[lot] = self.S[lot] + self.A[curr_k] * (input_sample.W[lot])

            # Calculate CT, LRT, TT
            self.CT[lot] = self.C[lot] - input_sample.A[lot]
            self.LRT[lot] = self.C[lot] - self.S[lot]
            self.TT[lot] = min(self.C[lot] - self.C[lot - 1], self.LRT[lot]) if lot != 0 else self.LRT[lot]

    def csv_write_params(self, filename):
        with open(filename, 'w', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(('Lot class', 'A'))
            for k, a in zip(range(len(self.A)), self.A):
                writer.writerow((k, a))

    def csv_write_run(self, filename):
        with open(filename, 'w', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(('Lot', 'S', 'C', 'CT', 'LRT', 'TT'))
            for lot, s, c, ct, lrt, tt in zip(range(len(self.S)), self.S, self.C, self.CT, self.LRT, self.TT):
                writer.writerow((lot, s, c, ct, lrt, tt))
