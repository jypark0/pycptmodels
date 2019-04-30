import numpy as np
import random

class Flowline:
  def __init__(self, 
              flow=
              [0, 1, 1, 1, 1, 2, 2, 3, 4, 3, 3, 3, 3, 2, 2, 0],
              R=
              [1, 2, 2, 2, 2, 1, 2, 2, 1, 2, 2, 2, 3, 2, 2, 1],
              PT=
              [0, 80, 90, 60, 65, 50, 90, 60, 100, 90, 60, 90, 130, 90, 60, 0],
              buffer_R=[1, 1, 16]):
    ''' Create Flowline model
    
    :param flow: Process flows. Must be a list for each lot class, where each list contains cluster indices
    Positive integer for each cluster with a robot arm, starting from left to right.
    0 for indexer
    :type flow: list of list

    :param R: Redundancies of process flow. Must be a list for each lot class
    :type R: list of list

    :param PT: Process times for process flow
    :type PT: list of list

    :param buffer_R: Buffer sizes, starting from left to right. Length of list must be equal to max cluster index - 1.
    :type buffer_R: list of int
    '''
    self.flow = flow
    self.R = R
    self.PT = PT
    self.buffer_R = buffer_R
  
  def Initialize(self):
    
    # Add buffers to flow, R, and PT
    m = 0
    for input_m in range(len(self.flow) - 1):
      # Check if m < m+1 (Prescan, different clusters)
      if self.flow[m] < self.flow[m+1]:
        # Check that m is not INDEXER and m+1 != 1
        if not (self.flow[m] == 0 and self.flow[m+1] == 1):
          # Prescan inlet buffer index = -1
          self.flow.insert(m+1, -1)

          # Get buffer size
          b = self.buffer_R[self.flow[m] - 1]
          self.R.insert(m+1, max(b - 1, 1))

          self.PT.insert(m+1, 0)
          
          # Increment to skip just added buffer
          m = m + 1
      
      # Check if m > m+1 (Post-scan, different clusters)
      if self.flow[m] > self.flow[m+1]:
        # Check that m != 1 and m+1 is not INDEXER
        if not (self.flow[m] == 1 and self.flow[m+1] == 0):
          # Post-scan outlet buffer index = -2
          self.flow.insert(m+1, -2)

          # Outlet buffer R = 1
          self.R.insert(m+1, 1)

          self.PT.insert(m+1, 0)
          
          # Increment to skip just added buffer
          m = m + 1
      m = m + 1

    # Modify PT
    

FL = Flowline()
FL.Initialize()