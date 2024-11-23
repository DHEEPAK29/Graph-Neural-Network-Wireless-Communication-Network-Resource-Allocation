import numpy as np
import matplotlib.pyplot as plt
import torch
from scipy.spatial import distance_matrix
from tqdm import tqdm

class Channel(object):
    def __init__(self, d0, gamma, s, N0):
        """
        Object for modeling a channel

        Args:
            d0: Reference distance.
            gamma: Path loss exponent.
            s: Fading energy.
            N0: Noise floor.
        """
        self.d0 = d0
        self.gamma = gamma
        self.s = s
        self.N0 = N0
    
    def pathloss(self, d):
        """
        Question 1.1
        Calculate simplified path-loss model.

        Args:
            d: The distance. Can be a matrix - this is an elementwise operation.

        Returns: Pathloss value.
        """
        return (self.d0 / d) ** self.gamma
    
    def fading_channel(self,d, Q):
        """
        Question 1.3
        Calculate fading channel model.

        Args:
            d: The distance. Can be a matrix - this is an elementwise operation.
            Q: Number of random samples.

        Returns: Q fading channel realizations
        """
        Exp_h = (self.d0 / d) ** self.gamma
        h_til = np.random.exponential(self.s, size=(1, Q))
        h = Exp_h * h_til / self.s
        return h
    
    def build_fading_capacity_channel(self, h, p):
        """
        Question 1.5
        Calculate fading capacity channel model.

        Args:
            h: Fading channel realizations (of length Q).
            p: Power values (of length Q).

        Returns: Q channel capacity values
        """
        return np.log(1 + h * p / self.N0)
