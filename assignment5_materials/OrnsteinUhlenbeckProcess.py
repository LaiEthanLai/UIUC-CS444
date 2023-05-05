import torch

class OrnsteinUhlenbeckProcess():
    def __init__(self, action_size, mu=0, sigma=1, dt=1e-2, theta=0.15) -> None:
        
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.action_size = action_size
        self.Nstep = 0
        self.theta = theta
        self.reset()

    def reset(self):
        self.prev_x = torch.zeros(self.action_size)

    def sample(self) -> torch.Tensor:
        current = self.prev_x + self.theta * (self.mu - self.prev_x) * self.dt + self.sigma * (self.dt ** 0.5) * torch.randn(self.action_size)
        self.prev_x = current
        self.Nstep += 1
        return current