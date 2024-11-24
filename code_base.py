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
        
class WirelessNetwork(object):
    def __init__(self, wx, wy, wc, n, d0, gamma, s, N0):
        """
        Object for modeling a Wireless Network

        Args:
            wx: Length of area.
            wy: Width of area.
            wc: Max distances of receiver from its transmitter.
            n: Number of transmitters/receivers.
            d0: Refernce distance.
            gamma: Pathloss exponent.
            s: Fading energy.
            N0: Noise floor.
        """
        self.wx = wx
        self.wy = wy
        self.wc = wc
        self.n = n

        # Determines transmitter and receiver positions
        self.t_pos, self.r_pos = self.determine_positions()

        # Calculate distance matrix using scipy.spatial method
        self.dist_mat = distance_matrix(self.t_pos, self.r_pos) 

        self.d0 = d0
        self.gamma = gamma
        self.s = s
        self.N0 = N0

        # Creates a channel with the given parameters
        self.channel = Channel(self.d0, self.gamma, self.s, self.N0)

    def determine_positions(self):
        """
        Question 2.1
        Calculate positions of transmitters and receivers

        Returns: transmitter positions, receiver positions
        """
        # Calculate transmitter positions
        t_x_pos = np.random.uniform(0, self.wx, (self.n, 1))
        t_y_pos = np.random.uniform(0, self.wy, (self.n, 1))
        t_pos = np.hstack((t_x_pos, t_y_pos))

        # Calculate receiver positions
        r_distance = np.random.uniform(0, self.wc, (self.n, 1))
        r_angle = np.random.uniform(0, 2 * np.pi, (self.n, 1))
        r_rel_pos = r_distance * np.hstack((np.cos(r_angle), np.sin(r_angle)))
        r_pos = t_pos + r_rel_pos
        return t_pos, r_pos
    
    def generate_pathloss_matrix(self):
        """
        Question 2.2
        Calculates pathloss matrix

        Returns: pathloss matrix
        """
        return self.channel.pathloss(self.dist_mat)

    def generate_interference_graph(self, Q):
        """
        Question 2.3
        Calculates interference graph

        Returns: interference graph
        """
        return self.channel.fading_channel(self.dist_mat, Q)

    def generate_channel_capacity(self, p, H):
        """
        Question 2.4
        Calculates capacity for each transmitter

        Returns: capacity for each transmitter
        """
        num = torch.diagonal(H, dim1=-2, dim2=-1) * p
        den = H.matmul(p.unsqueeze(-1)).squeeze() - num + self.N0
        return torch.log(1 + num / den)
    
    def plot_network(self):
        """
        Creates a plot of the given Wireless Network
        """
        plt.scatter(self.t_pos[:,0], self.t_pos[:,1], s = 4, label = "Transmitters")
        plt.scatter(self.r_pos[:,0], self.r_pos[:,1], s = 4, label = "Receivers", c = "orange")
        plt.xlabel("Area Length")
        plt.ylabel("Area Width")
        plt.title("Wireless Network")
        plt.savefig('WirelessNetwork.png', dpi = 200)
        plt.legend()
        return plt.show()
