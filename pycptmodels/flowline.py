import numpy as np
import random

from input import PoissonProcessInput

class ParametricFlowline:
  def __init__(self, 
              flow=[
              [1, 1, 1, 1, 1, 2, 2, 3, 4, 3, 3, 3, 3, 2, 2, 1],
              [1, 1, 1, 1, 1, 1, 2, 2, 4, 3, 3, 3, 3, 2, 2, 1],
              [1, 1, 1, 1, 2, 2, 2, 1, 2, 3, 4, 3, 3, 3, 3, 2, 2, 1],],
              R=[
              [1, 2, 2, 2, 2, 1, 2, 2, 1, 2, 2, 2, 3, 2, 2, 1],
              [1, 2, 2, 2, 2, 2, 2, 1, 1, 2, 2, 2, 3, 2, 2, 1],
              [1, 2, 2, 2, 1, 2, 2, 2, 2, 2, 1, 2, 2, 2, 3, 2, 2, 1],],
              PT=[
              [0, 80, 90, 60, 65, 50, 90, 60, 100, 90, 60, 90, 130, 90, 60, 0],
              [0, 80, 90, 60, 65, 90, 60, 50, 100, 90, 60, 90, 130, 90, 60, 0],
              [0, 80, 90, 60, 50, 90, 60, 65, 90, 60, 100, 90, 60, 90, 130, 90, 60, 0],],
              buffer_R=[4, 8, 16],
              move=3, 
              pick=1):
    ''' Create Flowline model
    
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
    '''
    # K = Number of lot classes
    self.K = len(flow)
    
    self.flow = flow
    self.R = R
    self.PT = PT
    self.buffer_R = buffer_R
    self.move = move
    self.pick = pick

  
  def initialize(self):
    ''' Create new process flows including buffers for parametric flow line. Modify process times
    '''
    
    # Add buffers to flow, R, and PT
    for k in range(self.K):

      m = 0
      orig_len = len(self.flow[k])
      for input_m in range(orig_len - 1):
        # Check if m < m+1, different clusters prescan case
        # Add as many buffers as the difference in clusters
        for c in range(self.flow[k][m], self.flow[k][m+1]):
          # Buffer flow = -1
          self.flow[k].insert(m+1, -1)

          # Get buffer size
          b = self.buffer_R[c - 1]
          self.R[k].insert(m+1, max(b - 1, 1))

          self.PT[k].insert(m+1, 0)

          # Increment to skip just added buffer
          m = m + 1

        # Check if m > m+1, different clusters postscan case
        # Add as many buffers as the difference in clusters
        for c in range(self.flow[k][m], self.flow[k][m+1], -1):
          # Buffer flow = -1
          self.flow[k].insert(m+1, -1)
          
          # Outlet buffer size = 1
          self.R[k].insert(m+1, 1)

          self.PT[k].insert(m+1, 0)
           
          # Increment to skip just added buffer
          m = m + 1

        m = m + 1

    # Add dummy modules to match maximum flow length
    max_len = max(len(f) for f in self.flow)
    self.dummy = [0] * self.K
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
    maxR = max([self.R[k][0] for k in range(self.K)])


    print("here")

FL = ParametricFlowline()
FL.initialize()

input = PoissonProcessInput(N=10, lambda_=2000, lotsizes=[23,24,25], lotsize_weights=[0.25,0.5,0.25], reticle=[210,260], prescan=[0,0], K=3)
input.initialize()

FL.run(input)
