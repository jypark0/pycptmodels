import numpy as np

from pycptmodels.input import PoissonProcessInput
from pycptmodels.fl import ParametricFlowLine


def test_initialize():
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

    assert np.array_equal(FL.flow, [
      [-2,-2,-2,-2,1,1,1,1,1,-1,2,2,-1,3,-1,4,-1,3,3,3,3,-1,2,2,-1,1],
      [-2,-2,-2,-2,1,1,1,1,1,1,-1,2,2,-1,-1,4,-1,3,3,3,3,-1,2,2,-1,1],
      [1,1,1,1,-1,2,2,2,-1,1,-1,2,-1,3,-1,4,-1,3,3,3,3,-1,2,2,-1,1],])
    assert np.array_equal(FL.R, [
      [1,1,1,1,1,2,2,2,2,1,1,2,1,2,15,1,1,2,2,2,3,1,2,2,1,1],
      [1,1,1,1,1,2,2,2,2,2,1,2,1,1,15,1,1,2,2,2,3,1,2,2,1,1],
      [1,2,2,2,1,1,2,2,1,2,1,2,1,2,15,1,1,2,2,2,3,1,2,2,1,1],])
    assert np.array_equal(FL.PT, [
      [0,0,0,0,5,85,95,65,70,5,63,95,5,65,5,110,5,95,65,95,135,5,95,65,5,0],
      [0,0,0,0,5,85,95,65,70,95,5,65,63,5,5,110,5,95,65,95,135,5,95,65,5,0],
      [5,85,95,65,5,63,95,65,5,70,5,95,5,65,5,110,5,95,65,95,135,5,95,65,5,0],])
    assert np.array_equal(FL.BN, [15, 15, 15])
    assert np.array_equal(FL.PBN, [10, 12, 5])
    assert np.array_equal(FL.dummy, [4, 4, 0])


def test_run():
    # Prescan test
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

    # Check shape of FL.X
    assert np.array(FL.X).shape == (25, 26)

    # Check last_prescan
    assert np.array_equal(FL.last_prescan, [13, 12, 13])

    # Start times
    assert np.array_equal(FL.S, [400., 585., 1915., 3120., 4495.])
    # Completion times
    assert np.array_equal(FL.C, [2323., 3123., 3923., 5213., 6418.])

    ## No Prescan test
    input2 = PoissonProcessInput(N=5, lambda_=0, lotsizes=[5], lotsize_weights=[
                                1], reticle=[250, 250], prescan=[0, 0], K=3)
    input2.initialize()

    # Fix arrival times and lot classes
    input2.lotclass = [1, 1, 0, 2, 0]
    input2.A = [0, 100, 150, 500, 600]

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
    FL.run(input)

    # Check shape of FL.X
    assert np.array(FL.X).shape == (25, 26)

    # Check last_prescan
    assert np.array_equal(FL.last_prescan, [13, 12, 13])

    # Start times
    assert np.array_equal(FL.S, [0., 185., 380., 500., 1115.])
    # Completion times
    assert np.array_equal(FL.C, [1923., 2723., 3523., 4323., 5123.])
