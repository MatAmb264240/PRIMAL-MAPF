import numpy as np


class ExperienceBuffer:
    def __init__(self, gamma: float = 0.99):
        self.gamma = gamma
        self.clear()

    def clear(self):
        self.states = []  # stan (fov, goal_vec, mask)
        self.actions = [] # akcję, którą wykonał
        self.rewards = [] # reward za ten krok
        self.values = []  # value, które przewidziała sieć
        self.dones = []   # info czy epizod się skończył

    def add(self, state, action: int, reward: float, value: float, done: bool):
        """
        Zapisuje pojedyncze przejście dla jednego agenta w jednym kroku.
        """
        self.states.append(state)
        self.actions.append(int(action))
        self.rewards.append(float(reward))
        self.values.append(float(value))
        self.dones.append(bool(done))

    def __len__(self):
        return len(self.states)

    def compute_returns_and_advantages(self):
        """
        Liczy zwroty G_t i advantage A_t = G_t - V(s_t).
        """
        n = len(self.rewards)
        returns = np.zeros(n, dtype=np.float32)

        running_return = 0.0
        for t in reversed(range(n)):
            running_return = self.rewards[t] + self.gamma * running_return * (1.0 - float(self.dones[t]))
            returns[t] = running_return

        values = np.array(self.values, dtype=np.float32)
        advantages = returns - values
        return returns, advantages