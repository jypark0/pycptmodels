import numpy as np

from pycptmodels.input import PoissonProcessInput


def test_initialize(flowline):
    FL = flowline
    FL.initialize()

    assert np.array_equal(FL.flow, [
        [-2, -2, -2, -2, 1, 1, 1, 1, 1, -1, 2, 2, -1, 3, -1, 4, -1, 3, 3, 3, 3, -1, 2, 2, -1, 1],
        [-2, -2, -2, -2, 1, 1, 1, 1, 1, 1, -1, 2, 2, -1, -1, 4, -1, 3, 3, 3, 3, -1, 2, 2, -1, 1],
        [1, 1, 1, 1, -1, 2, 2, 2, -1, 1, -1, 2, -1, 3, -1, 4, -1, 3, 3, 3, 3, -1, 2, 2, -1, 1], ])
    assert np.array_equal(FL.R, [
        [1, 1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 2, 1, 2, 15, 1, 1, 2, 2, 2, 3, 1, 2, 2, 1, 1],
        [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 1, 2, 1, 1, 15, 1, 1, 2, 2, 2, 3, 1, 2, 2, 1, 1],
        [1, 2, 2, 2, 1, 1, 2, 2, 1, 2, 1, 2, 1, 2, 15, 1, 1, 2, 2, 2, 3, 1, 2, 2, 1, 1], ])
    assert np.array_equal(FL.PT, [
        [0, 0, 0, 0, 5, 85, 95, 65, 70, 5, 63, 95, 5, 65, 5, 110, 5, 95, 65, 95, 135, 5, 95, 65, 5, 0],
        [0, 0, 0, 0, 5, 85, 95, 65, 70, 95, 5, 65, 63, 5, 5, 110, 5, 95, 65, 95, 135, 5, 95, 65, 5, 0],
        [5, 85, 95, 65, 5, 63, 95, 65, 5, 70, 5, 95, 5, 65, 5, 110, 5, 95, 65, 95, 135, 5, 95, 65, 5, 0], ])
    assert np.array_equal(FL.BN, [15, 15, 15])
    assert np.array_equal(FL.PBN, [10, 12, 5])
    assert np.array_equal(FL.dummy, [4, 4, 0])


def test_run_prescan(flowline):
    # Prescan test
    input1 = PoissonProcessInput(N=5, lambda_=0, lotsizes=[5], lotsize_weights=[
        1], reticle=[250, 250], prescan=[400, 400], K=3)
    input1.initialize()

    # Fix arrival times and lot classes
    input1.lotclass = [1, 1, 0, 2, 0]
    input1.A = [0, 500, 1500, 2500, 2600]

    FL = flowline
    FL.initialize()
    FL.run(input1)

    assert np.array(FL.X).shape == (25, 26)
    assert np.array_equal(FL.last_prescan, [13, 12, 13])

    assert np.array_equal(FL.L, [0., 405., 590., 1920., 3305.])
    assert np.array_equal(FL.S, [400., 585., 1915., 3120., 4495.])
    assert np.array_equal(FL.C, [2323., 3123., 3923., 5213., 6418.])

    # CT, LRT, TT
    assert np.array_equal(FL.CT, [2323., 2623., 2423., 2713., 3818.])
    assert np.array_equal(FL.LRT, [1923., 2538., 2008., 2093., 1923.])
    assert np.array_equal(FL.TT, [1923., 800., 800., 1290., 1205.])


def test_run_reticle(flowline):
    # No Prescan test
    input2 = PoissonProcessInput(N=5, lambda_=0, lotsizes=[5], lotsize_weights=[
        1], reticle=[250, 250], prescan=[0, 0], K=3)
    input2.initialize()

    # Fix arrival times and lot classes
    input2.lotclass = [1, 1, 0, 2, 0]
    input2.A = [0, 100, 150, 500, 600]

    FL = flowline
    FL.initialize()
    FL.run(input2)

    assert np.array(FL.X).shape == (25, 26)
    assert np.array_equal(FL.last_prescan, [13, 12, 13])

    assert np.array_equal(FL.L, [0., 5., 190., 475., 685.])
    assert np.array_equal(FL.S, [0., 185., 380., 500., 1115.])
    assert np.array_equal(FL.C, [1923., 2723., 3523., 4323., 5123.])

    # CT, LRT, TT
    assert np.array_equal(FL.CT, [1923., 2623., 3373., 3823., 4523.])
    assert np.array_equal(FL.LRT, [1923., 2538., 3143., 3823., 4008.])
    assert np.array_equal(FL.TT, [1923., 800., 800., 800., 800.])
