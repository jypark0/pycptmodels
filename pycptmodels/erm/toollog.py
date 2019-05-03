import numpy as np

from pycptmodels.input import PoissonProcessInput
from pycptmodels.fl import ParametricFlowLine

class ToolERM:
    def __init__(self):
        self.phi1 = []
        self.phi2 = []
        self.L_eq = []
        self.L_neq = []

        self.A1 = []
        self.B1 = []
        self.A2 = []
        self.B2 = []
        self.Dm = []
        self.Dp = []
        self.E = []

    def train(self, input_sample, X, S_l, C_l, C_w, BN, R, move, pick):
        A1_sum = [0] * input_sample.K
        A1_count = [0] * input_sample.K
        B1_sum = np.zeros((input_sample.K, input_sample.K)).tolist()
        B1_count = np.zeros((input_sample.K, input_sample.K)).tolist()

        A2_sum = [0] * input_sample.K
        A2_count = [0] * input_sample.K
        B2_sum = np.zeros((input_sample.K, input_sample.K)).tolist()
        B2_count = np.zeros((input_sample.K, input_sample.K)).tolist()

        Dm_sum = [0] * input_sample.K
        Dm_count = [0] * input_sample.K
        Dp_sum = [0] * input_sample.K
        Dp_count = [0] * input_sample.K

        E_sum = np.zeros((input_sample.K, input_sample.K)).tolist()
        E_count = np.zeros((input_sample.K, input_sample.K)).tolist()

        # Discard first and last lots
        for lot in range(1, input_sample.N - 1):

            curr_k = input_sample.lotclass[lot]
            prev_k = input_sample.lotclass[lot - 1]
            next_k = input_sample.lotclass[lot + 1]

            LHS = X[input_sample.first_wfr_idx[lot]][BN[curr_k]]
            RHS = X[input_sample.first_wfr_idx[lot] - 1][BN[curr_k] + 1] + move + 2 * pick

            # No bottleneck contention
            if LHS > RHS:
                self.phi1.append(lot)

                A1_sum[curr_k] += (C_l[lot] - C_w[input_sample.first_wfr_idx[lot]])
                A1_count[curr_k] += input_sample.W[lot] - 1

                B1_sum[curr_k][prev_k] += (C_w[input_sample.first_wfr_idx[lot]] - S_l[lot])
                B1_count[curr_k][prev_k] += ()

                # TODO - finish train function

            else:

                if LHS > RHS:
                    self.phi1.append(lot)
                    NBC = True
                else:
                    self.phi2.append(lot)

            if NBC:
                self.A1[curr_k]

            # Calculate L_eq and L_neq lot indices (exclude last lot)
            if lot != input_sample.N - 1:
                if input_sample.lotclass[lot] == input_sample.lotclass[lot+1]:
                    self.L_eq.append(lot)
                else:
                    self.L_neq.append(lot)

        # Calculate parameters
        self.A1 = [0] * self.K
        self.B1 = [0] * self.K
        self.A2 = [0] * self.K
        self.B2 = np.zeros((input_sample.K, input_sample.K)).tolist()
        for k1 in range(self.K):
            self.A1[k1] = np.sum(C_l[self.phi1])


        print("here")


input1 = PoissonProcessInput(N=5, lambda_=0, lotsizes=[5], lotsize_weights=[
                            1], reticle=[250, 250], prescan=[400, 400], K=3)
input1.initialize()

# Fix arrival times and lot classes
input1.lotclass = [1, 1, 0, 2, 0]
input1.A = [0, 500, 1500, 2500, 2600]

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

erm3 = ToolERM()
erm3.train(input1, FL.X, FL.S, FL.C, FL.C_w, FL.BN, FL.R, FL.move, FL.pick)

