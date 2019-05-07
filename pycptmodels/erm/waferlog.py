import numpy as np

from pycptmodels.input import PoissonProcessInput
from pycptmodels.fl import ParametricFlowLine


class WaferERM:
    def __init__(self):
        self.phi = []
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

    # TODO: create unit test
    def train(self, input_sample, L_l, S_l, C_l, S_w, C_w, R):
        self.phi = np.zeros(input_sample.N, dtype=int).tolist()

        self.A1 = np.zeros(input_sample.K).tolist()
        A1_sum = np.zeros(input_sample.K).tolist()
        A1_count = np.zeros(input_sample.K, dtype=int).tolist()
        self.B1 = np.zeros((input_sample.K, input_sample.K)).tolist()
        B1_sum = np.zeros((input_sample.K, input_sample.K)).tolist()
        B1_count = np.zeros((input_sample.K, input_sample.K), dtype=int).tolist()

        self.A2 = np.zeros(input_sample.K).tolist()
        A2_sum = np.zeros(input_sample.K).tolist()
        A2_count = np.zeros(input_sample.K, dtype=int).tolist()
        self.B2 = np.zeros(input_sample.K).tolist()
        B2_sum = np.zeros(input_sample.K).tolist()
        B2_count = np.zeros(input_sample.K, dtype=int).tolist()

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
            prev_k = input_sample.lotclass[lot-1]

            # No bottleneck contention
            if S_l[lot] > C_l[lot-1]:
                self.phi[lot] = 1
                self.phi1.append(lot)

                A1_sum[curr_k] += (C_l[lot] - C_w[input_sample.first_wfr_idx[lot]])
                A1_count[curr_k] += (input_sample.W[lot] - 1)

                B1_sum[curr_k][prev_k] += (C_w[input_sample.first_wfr_idx[lot]] - S_l[lot])
                B1_count[curr_k][prev_k] += 1

            # Bottleneck contention
            if curr_k == prev_k and input_sample.A[lot] <= S_w[input_sample.first_wfr_idx[lot] - 1]:
                self.phi[lot] = 2
                self.phi2.append(lot)

                A2_sum[curr_k] += (C_l[lot] - C_w[input_sample.first_wfr_idx[lot]])
                A2_count[curr_k] += (input_sample.W[lot] - 1)

                B2_sum[curr_k] += (C_w[input_sample.first_wfr_idx[lot]] - C_l[lot-1])
                B2_count[curr_k] += 1

            # E
            E_sum[curr_k][prev_k] += (S_l[lot] - L_l[lot])
            E_count[curr_k][prev_k] += 1

        # Check if last lot phi1 or phi2
        if S_l[-1] > C_l[-2]:
            self.phi[lot] = 1
            self.phi1.append(input_sample.N - 1)
        if input_sample.lotclass[-1] == input_sample.lotclass[-2] and input_sample.A[-1] <= S_w[input_sample.first_wfr_idx[-1] - 1]:
            self.phi[lot] = 2
            self.phi2.append(input_sample.N - 1)

        # Calculate vacation time related parameters
        for lot in range(1, input_sample.N - 1):
            curr_k = input_sample.lotclass[lot]

            if self.phi[lot + 1] == 2:
                Dm_sum[curr_k] += (C_l[lot] - L_l[lot + 1])
                Dm_count[curr_k] += 1

        # Average all parameters
        for k1 in range(input_sample.K):
            self.A1[k1] = A1_sum[k1] / A1_count[k1] if A1_count[k1] else 0.
            self.A2[k1] = A2_sum[k1] / A2_count[k1] if A2_count[k1] else 0.
            self.Dm[k1] = Dm_sum[k1] / Dm_count[k1] if Dm_count[k1] else 0.
            self.Dp[k1] = self.Dm[k1] - (R[k1][0] - 1) * self.A2[k1]
            self.B2[k1] = B2_sum[k1] / B2_count[k1] if B2_count[k1] else 0.

            for k2 in range(input_sample.K):
                self.B1[k1][k2] = B1_sum[k1][k2] / B1_count[k1][k2] if B1_count[k1][k2] else 0.
                self.E[k1][k2] = E_sum[k1][k2] / E_count[k1][k2] if E_count[k1][k2] else 0.


    # TODO: Check if correct
    def run(self, input_sample):
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
                prev_k = (curr_k+1) % input_sample.K
                self.L[lot] = input_sample.A[lot]
                self.S[lot] = self.L[lot]
            else:
                prev_k = input_sample.lotclass[lot-1]
                if curr_k == prev_k:
                    self.V[lot-1] = self.Vm[lot-1]
                else:
                    self.V[lot-1] = self.Vp[lot-1]
                self.L[lot] = max(input_sample.A[lot], self.V[lot-1])
                self.S[lot] = self.L[lot] + self.E[curr_k][prev_k]

            self.C[lot] = max(self.S[lot] + self.B1[curr_k][prev_k] + self.A1[curr_k]*(input_sample.W[lot] - 1),
                              self.C[lot-1] + self.B2[curr_k][prev_k] + self.A2[curr_k]*(input_sample.W[lot] - 1))
            self.Vm[lot] = self.C[lot] - self.Dm[curr_k]
            self.Vp[lot] = self.C[lot] - self.Dp[curr_k]

            self.CT[lot] = self.S[lot] - input_sample.A[lot]
            self.LRT[lot] = self.C[lot] - self.S[lot]
            self.TT[lot] = min(self.C[lot] - self.C[lot - 1], self.LRT[lot]) if lot != 0 else self.LRT[lot]


input1 = PoissonProcessInput(N=1000, lambda_=3300, lotsizes=[25], lotsize_weights=[
                            1], reticle=[250, 250], prescan=[400, 400], K=3)
input1.initialize()

FL = ParametricFlowLine(
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
  pick=1
)
FL.initialize()
FL.run(input1)

erm = WaferERM()
erm.train(input1, FL.L, FL.S, FL.C, FL.S_w, FL.C_w, FL.R)
print("here")