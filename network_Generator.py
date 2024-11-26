class Generator:
    def __init__(self, n, wx, wy, wc, d0=1, gamma=2.2, s=2, N0=1, device="cpu", batch_size=64, random=False):
        # Save the configurations for the Wireless Network
        self.n = n
        self.wx = wx
        self.wy = wy
        self.wc = wc

        # Save the Channel configurations
        self.d0 = d0
        self.gamma = gamma
        self.s = s
        self.N0 = N0

        # Training configurations
        self.device = device
        self.batch_size = batch_size

        # True if pathloss should change at random
        self.random = random

        self.train = None
        self.test = None

        # Generate a Wireless Network and pathloss matrix
        self.network = WirelessNetwork(self.wx, self.wy, self.wc, self.n, self.d0, 
                                       self.gamma, self.s, self.N0)
        self.H1 = self.network.generate_pathloss_matrix()
        
    def __next__(self):
        if self.random:
            # Generate a new random network
            self.network = WirelessNetwork(self.wx, self.wy, self.wc, self.n, self.d0, 
                                       self.gamma, self.s, self.N0)
            self.H1 = self.network.generate_pathloss_matrix()
        H2 = np.random.exponential(self.s, (self.batch_size, self.n, self.n))

        # Generate random channel matrix
        H = self.H1 * H2

        # Normalization of the channel matrix
        eigenvalues, _ = np.linalg.eig(H)
        S = H / np.max(eigenvalues.real)

        # Put onto device
        H = torch.from_numpy(H).to(torch.float).to(self.device)
        S = torch.from_numpy(S).to(torch.float).to(self.device)
        return H, S, self.network
