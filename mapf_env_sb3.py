# mapf_env_sb3.py
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from mapf_env import SimpleMAPFEnv


class MAPF_SB3Env(gym.Env):
    """
    Wrapper na SimpleMAPFEnv pod Stable-Baselines3.

    - obs: spłaszczony wektor wszystkich agentów
    - action_space: MultiDiscrete([5] * num_agents)
    - reward: suma nagród agentów (team reward)
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        grid_size: int = 10,
        num_agents: int = 4,
        fov_size: int = 10,
        obstacle_density: float = 0.1,
        max_steps: int = 64,
    ):
        super().__init__()
        self.grid_size = grid_size
        self.num_agents = num_agents
        self.fov_size = fov_size
        self.obstacle_density = obstacle_density
        self.max_steps = max_steps

        self._env = SimpleMAPFEnv(
            grid_size=self.grid_size,
            num_agents=self.num_agents,
            fov_size=self.fov_size,
            obstacle_density=self.obstacle_density,
            max_steps=self.max_steps,
        )

        self.action_space = spaces.MultiDiscrete([5] * self.num_agents)

        fov_dim = self.fov_size * self.fov_size * 4
        goal_dim = 3
        mask_dim = 5
        per_agent_dim = fov_dim + goal_dim + mask_dim
        obs_dim = per_agent_dim * self.num_agents

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )

        self._last_info = {}

    def _flatten_obs_list(self, obs_list):
        per_agent_vecs = []
        for obs in obs_list:  # Access the observations for all agents
            fov = obs["fov"].astype(np.float32).reshape(-1)
            goal_vec = obs["goal_vec"].astype(np.float32).reshape(-1)
            action_mask = obs["action_mask"].astype(np.float32).reshape(-1)
            vec = np.concatenate([fov, goal_vec, action_mask], axis=0)
            per_agent_vecs.append(vec)

        flat = np.concatenate(per_agent_vecs, axis=0)
        return flat
    
    def set_obstacle_density(self, density: float):
        self.obstacle_density = float(density)
        self._env.set_obstacle_density(density)

    def reset(self, seed=None, options=None):
        if seed is not None:
            self._env.seed(seed)

        obs_list, _ = self._env.reset()  # Correct the output variable name
        flat_obs = self._flatten_obs_list(obs_list)

        info = {
            "num_agents_at_goal": 0,
            "pct_agents_at_goal": 0.0,
        }
        return flat_obs, info

    def step(self, action):
        """
        action: np.ndarray [num_agents] (MultiDiscrete)
        """
        action = np.array(action, dtype=np.int32).tolist()

        obs_list, rewards, done, info = self._env.step(action)
        self._last_info = info

        flat_obs = self._flatten_obs_list(obs_list)

        # team reward
        reward_team = float(sum(rewards))

        terminated = bool(done)
        truncated = bool(self._env.steps >= self.max_steps)

        return flat_obs, reward_team, terminated, truncated, info

    def render(self):
        self._env.debug_print_map()

