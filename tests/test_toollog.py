import numpy as np

from pycptmodels.erm import ToolERM
from pycptmodels.fl import ParametricFlowLine
from pycptmodels.input import PoissonProcessInput


def test_train():
    input1 = PoissonProcessInput(N=100, lambda_=1000, lotsizes=[5], lotsize_weights=[
        1], reticle=[200, 200], prescan=[400, 400], K=3)
    input1.initialize()
    input1.A = [0, 100, 300, 1200, 1300, 1800, 2100, 2500, 3300, 4300, 6100, 7200, 11700, 12100, 12300, 12500, 13100,
                13500, 13700, 13900, 14100, 14700, 16000, 16600, 17500, 18200, 20800, 22200, 23000, 23700, 24000, 24800,
                26000, 28700, 29900, 30600, 31300, 31500, 34100, 34800, 35000, 38900, 39200, 39300, 39600, 40300, 47200,
                47400, 49400, 51900, 53300, 54000, 55200, 58100, 60500, 60600, 60900, 61300, 61400, 62300, 64300, 66600,
                67500, 68700, 70100, 70600, 75700, 77200, 77900, 77900, 78000, 78200, 78900, 78900, 79800, 80300, 80600,
                81100, 81700, 82000, 82100, 82500, 82700, 83100, 83300, 83500, 83600, 83800, 83900, 84900, 85900, 86000,
                88700, 89800, 89900, 91500, 94100, 95000, 96000, 96300]
    input1.W = [5] * input1.N
    input1.lotclass = [1, 2, 0, 0, 1, 0, 0, 2, 0, 1, 2, 0, 0, 2, 1, 2, 0, 1, 2, 1, 2, 1, 1, 0, 1, 0, 0, 0, 1, 2, 1, 0,
                       1, 2, 2, 0, 1, 2, 2, 0, 2, 2, 1, 2, 0, 1, 0, 1, 1, 1, 0, 2, 1, 1, 1, 1, 2, 2, 0, 1, 0, 2, 0, 0,
                       2, 1, 0, 2, 2, 2, 2, 0, 1, 0, 2, 2, 2, 0, 2, 2, 0, 0, 0, 2, 1, 0, 2, 0, 1, 2, 0, 0, 1, 2, 0, 1,
                       1, 1, 1, 2]

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
    erm3.train(input1, FL.X, FL.L, FL.S, FL.C, FL.C_w, FL.BN, FL.R, FL.move, FL.pick)

    assert np.array_equal(erm3.phi1,
                          [1, 2, 4, 5, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 23, 24, 25, 29, 30, 31, 32,
                           33, 35, 36, 37, 39, 40, 42, 43, 44, 45, 46, 47, 50, 51, 52, 53, 54, 56, 58, 59, 60, 61, 62,
                           64, 65, 66, 67, 73, 74, 78, 80, 84, 85, 86, 87, 88, 89, 90, 92, 93, 94, 95])
    assert np.array_equal(erm3.A1, [110., 110., 110.])
    assert np.array_equal(erm3.B1, [1433., 1433., 1603.])

    assert np.array_equal(erm3.phi2,
                          [3, 6, 12, 22, 26, 27, 28, 34, 38, 41, 48, 49, 55, 57, 63, 68, 69, 70, 71, 72, 75, 76, 77, 79,
                           81, 82, 83, 91, 96, 97, 98])
    assert np.array_equal(erm3.A2, [110., 110., 110.])
    assert np.array_equal(erm3.B2, [[310., 0., 310.], [310., 310., 0.], [310., 0., 310.]])

    assert np.array_equal(erm3.L_eq,
                          [2, 5, 11, 21, 25, 26, 33, 37, 40, 47, 48, 52, 53, 54, 56, 62, 67, 68, 69, 74, 75, 78, 80, 81,
                           90, 95, 96, 97])
    assert np.array_equal(erm3.L_neq,
                          [1, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 22, 23, 24, 27, 28, 29, 30, 31,
                           32, 34, 35, 36, 38, 39, 41, 42, 43, 44, 45, 46, 49, 50, 51, 55, 57, 58, 59, 60, 61, 63, 64,
                           65, 66, 70, 71, 72, 73, 76, 77, 79, 82, 83, 84, 85, 86, 87, 88, 89, 91, 92, 93, 94, 98])
    assert np.allclose(erm3.Dm, [1993.55556, 2108.555556, 2057.3])
    assert np.allclose(erm3.Dp, [2114.4, 2055.5, 2094])

    assert np.allclose(erm3.E, [[0., 1389.5, 868.], [1048.214286, 486.111111, 868.], [1303.636364, 1205.833333, 11.5]])
