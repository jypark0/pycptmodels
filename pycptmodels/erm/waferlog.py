import numpy as np

from pycptmodels.input import PoissonProcessInput
from pycptmodels.fl import ParametricFlowLine


class WaferERM:
    def __init__(self):
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
        phi = np.zeros(input_sample.N, dtype=int).tolist()

        self.A1 = np.zeros(input_sample.K).tolist()
        A1_sum = np.zeros(input_sample.K).tolist()
        A1_count = np.zeros(input_sample.K, dtype=int).tolist()
        self.B1 = np.zeros(input_sample.K).tolist()
        B1_sum = np.zeros(input_sample.K).tolist()
        B1_count = np.zeros(input_sample.K, dtype=int).tolist()

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
            prev_k = input_sample.lotclass[lot - 1]

            # No bottleneck contention
            if S_l[lot] > C_l[lot - 1]:
                phi[lot] = 1
                self.phi1.append(lot)

                A1_sum[curr_k] += (C_l[lot] - C_w[input_sample.first_wfr_idx[lot]])
                A1_count[curr_k] += (input_sample.W[lot] - 1)

                B1_sum[curr_k] += (C_w[input_sample.first_wfr_idx[lot]] - S_l[lot])
                B1_count[curr_k] += 1

            # Bottleneck contention
            elif curr_k == prev_k and input_sample.A[lot] <= S_w[input_sample.first_wfr_idx[lot] - 1]:
                phi[lot] = 2
                self.phi2.append(lot)

                A2_sum[curr_k] += (C_l[lot] - C_w[input_sample.first_wfr_idx[lot]])
                A2_count[curr_k] += (input_sample.W[lot] - 1)

                B2_sum[curr_k] += (C_w[input_sample.first_wfr_idx[lot]] - C_l[lot - 1])
                B2_count[curr_k] += 1

            # E
            E_sum[curr_k][prev_k] += (S_l[lot] - L_l[lot])
            E_count[curr_k][prev_k] += 1

        # Check if last lot phi1 or phi2
        if S_l[-1] > C_l[-2]:
            phi[lot] = 1
            self.phi1.append(input_sample.N - 1)
        elif input_sample.lotclass[-1] == input_sample.lotclass[-2] and \
                input_sample.A[-1] <= S_w[input_sample.first_wfr_idx[-1] - 1]:
            phi[lot] = 2
            self.phi2.append(input_sample.N - 1)

        # Calculate vacation time related parameters
        for lot in range(1, input_sample.N - 1):
            curr_k = input_sample.lotclass[lot]

            if phi[lot + 1] == 2:
                Dm_sum[curr_k] += (C_l[lot] - L_l[lot + 1])
                Dm_count[curr_k] += 1

        # Average all parameters
        for k1 in range(input_sample.K):
            self.A1[k1] = A1_sum[k1] / A1_count[k1] if A1_count[k1] else 0.
            self.A2[k1] = A2_sum[k1] / A2_count[k1] if A2_count[k1] else 0.
            self.B1[k1] = B1_sum[k1] / B1_count[k1] if B1_count[k1] else 0.
            self.B2[k1] = B2_sum[k1] / B2_count[k1] if B2_count[k1] else 0.
            self.Dm[k1] = Dm_sum[k1] / Dm_count[k1] if Dm_count[k1] else 0.
            self.Dp[k1] = self.Dm[k1] - (R[k1][0] - 1) * self.A2[k1]

            for k2 in range(input_sample.K):
                self.E[k1][k2] = E_sum[k1][k2] / E_count[k1][k2] if E_count[k1][k2] else 0.

        print("here")

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


input1 = PoissonProcessInput(N=200, lambda_=1500, lotsizes=[5], lotsize_weights=[
    1], reticle=[200, 200], prescan=[400, 400], K=3)
input1.initialize()
input1.A = [0, 100, 900, 8100, 8500, 11900, 12100, 19100, 20100, 20800, 20800, 24300, 24600, 25400, 27700, 29900, 30100,
            30300, 31300, 33200, 37800, 38900, 39600, 40800, 41000, 42000, 44900, 46300, 47300, 50200, 51500, 52300,
            52900, 53900, 61300, 62200, 62400, 64900, 66100, 67500, 71700, 72000, 72800, 75400, 75600, 75900, 77100,
            77300, 77400, 80000, 81900, 81900, 82600, 85700, 86300, 87000, 89200, 95300, 95500, 95800, 97800, 99500,
            101000, 104800, 105700, 108900, 108900, 110000, 111000, 112800, 113300, 114600, 114600, 115900, 119900,
            120700, 121100, 126100, 130100, 138600, 139500, 139600, 140200, 143000, 144300, 144700, 146500, 148400,
            150000, 151000, 151600, 151900, 152700, 154000, 155700, 156100, 160100, 160300, 160500, 161700, 163000,
            164400, 164700, 172400, 172400, 174600, 179500, 180300, 181100, 182500, 184500, 186100, 186800, 190000,
            190000, 191600, 193300, 194100, 199100, 199700, 200600, 203300, 204700, 206800, 207100, 208700, 210300,
            216500, 218400, 225600, 226000, 226900, 227900, 228000, 229400, 230300, 231200, 232700, 237100, 239700,
            241200, 243000, 243700, 245500, 246400, 248500, 248700, 248900, 249800, 250300, 250600, 250900, 252700,
            256700, 257000, 257400, 257700, 266200, 267200, 267900, 272200, 272700, 273800, 275900, 277700, 278900,
            280700, 280900, 282300, 283000, 285300, 285600, 286400, 287600, 289300, 290100, 292000, 292100, 292200,
            292300, 293000, 293100, 293300, 293300, 296300, 296500, 297700, 302700, 303600, 303900, 304000, 305600,
            308500, 308700, 311100, 311900, 314500, 315300, 315300, 315600]
input1.lotclass = [0, 2, 0, 0, 2, 0, 1, 2, 0, 1, 1, 0, 2, 1, 1, 1, 2, 0, 2, 1, 1, 1, 2, 2, 0, 1, 1, 0, 2, 0, 0, 0, 2, 2,
                   0, 2, 2, 0, 1, 0, 1, 0, 2, 0, 0, 2, 0, 2, 0, 2, 2, 1, 1, 1, 2, 2, 0, 0, 2, 0, 1, 2, 2, 1, 1, 2, 1, 1,
                   1, 2, 1, 0, 0, 2, 0, 2, 0, 0, 2, 1, 1, 2, 0, 2, 1, 0, 1, 1, 0, 0, 0, 2, 0, 2, 2, 1, 2, 2, 2, 0, 2, 0,
                   0, 1, 1, 2, 1, 0, 1, 2, 1, 2, 0, 2, 1, 1, 1, 1, 1, 0, 1, 1, 2, 0, 1, 0, 1, 1, 0, 2, 2, 0, 1, 1, 0, 0,
                   1, 2, 1, 1, 0, 1, 0, 2, 2, 0, 1, 1, 1, 0, 0, 0, 2, 2, 0, 1, 1, 2, 1, 2, 0, 2, 2, 1, 2, 1, 2, 2, 2, 1,
                   0, 1, 1, 1, 2, 1, 2, 2, 2, 2, 2, 1, 1, 0, 0, 0, 2, 1, 0, 2, 0, 1, 1, 2, 1, 1, 2, 2, 1, 2]

# input1.A = [0, 100, 300, 1200, 1300, 1800, 2100, 2500, 3300, 4300, 6100, 7200, 11700, 12100, 12300, 12500, 13100,
#             13500, 13700, 13900, 14100, 14700, 16000, 16600, 17500, 18200, 20800, 22200, 23000, 23700, 24000, 24800,
#             26000, 28700, 29900, 30600, 31300, 31500, 34100, 34800, 35000, 38900, 39200, 39300, 39600, 40300, 47200,
#             47400, 49400, 51900, 53300, 54000, 55200, 58100, 60500, 60600, 60900, 61300, 61400, 62300, 64300, 66600,
#             67500, 68700, 70100, 70600, 75700, 77200, 77900, 77900, 78000, 78200, 78900, 78900, 79800, 80300, 80600,
#             81100, 81700, 82000, 82100, 82500, 82700, 83100, 83300, 83500, 83600, 83800, 83900, 84900, 85900, 86000,
#             88700, 89800, 89900, 91500, 94100, 95000, 96000, 96300]
# input1.lotclass = [1, 2, 0, 0, 1, 0, 0, 2, 0, 1, 2, 0, 0, 2, 1, 2, 0, 1, 2, 1, 2, 1, 1, 0, 1, 0, 0, 0, 1, 2, 1, 0,
#                    1, 2, 2, 0, 1, 2, 2, 0, 2, 2, 1, 2, 0, 1, 0, 1, 1, 1, 0, 2, 1, 1, 1, 1, 2, 2, 0, 1, 0, 2, 0, 0,
#                    2, 1, 0, 2, 2, 2, 2, 0, 1, 0, 2, 2, 2, 0, 2, 2, 0, 0, 0, 2, 1, 0, 2, 0, 1, 2, 0, 0, 1, 2, 0, 1,
#                    1, 1, 1, 2]
# input1.csv_write('input.csv')

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
# FL.csv_write_lot('flowline_lot.csv')
# FL.csv_write_wfr('flowline_wfr.csv')

erm = WaferERM()
erm.train(input1, FL.L, FL.S, FL.C, FL.S_w, FL.C_w, FL.R)
print("here")
