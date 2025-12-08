import torch as th
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class MAPFFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box,
                 num_agents: int = 4, fov_size: int = 10, n_channels: int = 4):
        super().__init__(observation_space, features_dim=256)

        self.num_agents = num_agents
        self.fov_size = fov_size
        self.n_channels = n_channels

        obs_dim = observation_space.shape[0]

        fov_dim = fov_size * fov_size * n_channels
        goal_dim = 3
        mask_dim = 5  # 5 actions

        self.per_agent_dim = fov_dim + goal_dim + mask_dim
        assert self.per_agent_dim * num_agents == obs_dim, (
            f"Obs dim {obs_dim} != per_agent_dim {self.per_agent_dim} * num_agents {num_agents}"
        )

        # CNN tower (6 convolutions + 2x MaxPool)
        self.conv1 = nn.Conv2d(n_channels, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 128, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2)

        self.conv4 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv5 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv6 = nn.Conv2d(256, 256, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2)

        # LayerNorm only on final convolution
        self.ln_final = nn.LayerNorm([256, fov_size // 2, fov_size // 2])

        conv_out_dim = 256 * 2 * 2
        self.fov_fc = nn.Linear(conv_out_dim, 256)
        self.goal_fc = nn.Linear(goal_dim, 32)

        # MLP
        self.final_fc = nn.Linear(256 + 32, 256)

        self.relu = nn.ReLU()
        self._features_dim = 256

    @property
    def features_dim(self) -> int:
        return self._features_dim

    def forward(self, obs: th.Tensor) -> th.Tensor:
        B = obs.size(0)

        obs = obs.view(B, self.num_agents, self.per_agent_dim)

        fov = obs[:, :, : self.fov_size * self.fov_size * self.n_channels]
        goal = obs[:, :, self.fov_size * self.fov_size * self.n_channels:
                        self.fov_size * self.fov_size * self.n_channels + 3]

        fov = fov.reshape(B * self.num_agents,
                          self.fov_size,
                          self.fov_size,
                          self.n_channels)
        x = fov.permute(0, 3, 1, 2)  # (BN, C, H, W)

        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.pool1(x)

        x = self.relu(self.conv4(x))
        x = self.relu(self.conv5(x))
        x = self.relu(self.ln_final(self.conv6(x)))
        x = self.pool2(x)

        x = th.flatten(x, 1)
        x = self.relu(self.fov_fc(x))

        # goal vector
        g = goal.reshape(B * self.num_agents, 3)
        g = self.relu(self.goal_fc(g))

        h = th.cat([x, g], dim=1)
        h = self.relu(self.final_fc(h))

        h = h.view(B, self.num_agents, 256)
        h = h.sum(dim=1)

        return h
