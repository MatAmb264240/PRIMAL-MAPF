# mapf_env_primal.py
import numpy as np
import random
import gymnasium as gym
from gymnasium import spaces
from collections import deque


class SimpleMAPFEnv(gym.Env):
    """
    MAPF w stylu PRIMAL:
      - brak shaping
      - brak nagród za wejście na cel
      - brak nagród za stanie na celu
      - jedyna duża nagroda = +20 przy sukcesie wszystkich agentów
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        grid_size=8,
        num_agents=3,
        fov_size=10,
        obstacle_density=0.1,
        max_steps=64,
    ):
        super().__init__()
        self.GRID_SIZE = grid_size
        self.NUM_AGENTS = num_agents
        self.FOV_SIZE = fov_size
        self.OBSTACLE_DENSITY = obstacle_density
        self.max_steps = max_steps
        self.seed_value = None  # Initialize the seed variable

        # 0 = stay, 1 = right, 2 = left, 3 = down, 4 = up
        self.ACTIONS = [(0, 0), (0, 1), (0, -1), (1, 0), (-1, 0)]

        # ---- PRIMAL REWARDS (Table I) ----
        self.R_MOVE = -0.3            # każdy ruch
        self.R_STAY_OFF = -0.5        # stanie poza celem
        self.R_STAY_ON = 0.0          # stanie na celu
        self.R_COLLISION = -2.0       # kolizja (vertex lub edge)
        self.R_SUCCESS = +20.0        # wszyscy dotarli – epizod zakończony
        # -----------------------------------

        self.action_space = spaces.Discrete(len(self.ACTIONS))
        self.observation_space = spaces.Dict(
            {
                "fov": spaces.Box(low=0, high=1, shape=(fov_size, fov_size, 4), dtype=np.float32),
                "goal_vec": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
                "action_mask": spaces.Box(low=0, high=1, shape=(len(self.ACTIONS),), dtype=np.float32),
            }
        )

        self.reset()
    def seed(self, seed=None):
        """
        Set the seed for the environment.
        """
        self.seed_value = seed
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    def _reachable(self, start, goal):
        """Zwraca True, jeśli istnieje ścieżka start→goal omijająca przeszkody."""
        (sx, sy) = start
        (gx, gy) = goal

        if self.OBSTACLES[sx, sy] == 1 or self.OBSTACLES[gx, gy] == 1:
            return False
        if (sx, sy) == (gx, gy):
            return True

        visited = set()
        queue = [(sx, sy)]
        visited.add((sx, sy))

        moves = [(1,0), (-1,0), (0,1), (0,-1)]

        while queue:
            x, y = queue.pop(0)

            for dx, dy in moves:
                nx, ny = x + dx, y + dy

                if not (0 <= nx < self.GRID_SIZE and 0 <= ny < self.GRID_SIZE):
                    continue

                if self.OBSTACLES[nx, ny] == 1:
                    continue

                if (nx, ny) in visited:
                    continue

                if (nx, ny) == (gx, gy):
                    return True

                visited.add((nx, ny))
                queue.append((nx, ny))

        return False

    # -------------------------------------------------------
    # RESET
    # -------------------------------------------------------
    def get_cbs_description(self):
        """
        Return a description of the environment for CBS.
        """
        return {
            "grid_size": self.GRID_SIZE,
            "obstacles": self.OBSTACLES.tolist(),
            "starts": self.agent_positions,
            "goals": self.agent_goals
        }
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0

        while True:
            # 1) generuj przeszkody
            self.OBSTACLES = np.zeros((self.GRID_SIZE, self.GRID_SIZE), dtype=np.int32)
            for x in range(self.GRID_SIZE):
                for y in range(self.GRID_SIZE):
                    if random.random() < self.OBSTACLE_DENSITY:
                        self.OBSTACLES[x, y] = 1

            # 2) generuj starty i cele
            occupied = set()
            starts = []
            goals = []

            valid = True

            for _ in range(self.NUM_AGENTS):
                # start
                for _try in range(100):
                    sx = random.randint(0, self.GRID_SIZE - 1)
                    sy = random.randint(0, self.GRID_SIZE - 1)
                    if self.OBSTACLES[sx, sy] == 0 and (sx, sy) not in occupied:
                        starts.append((sx, sy))
                        occupied.add((sx, sy))
                        break
                else:
                    valid = False
                    break

                # goal
                for _try in range(100):
                    gx = random.randint(0, self.GRID_SIZE - 1)
                    gy = random.randint(0, self.GRID_SIZE - 1)
                    if self.OBSTACLES[gx, gy] == 0 and (gx, gy) not in occupied:
                        goals.append((gx, gy))
                        occupied.add((gx, gy))
                        break
                else:
                    valid = False
                    break

            if not valid:
                continue

            # 3) SPRAWDŹ osiągalność
            reachable = True
            for s, g in zip(starts, goals):
                if not self._reachable(s, g):
                    reachable = False
                    break

            if reachable:
                break  # mapa jest dobra – wychodzimy z while True

        self.agent_positions = starts
        self.agent_goals = goals
        self.agent_reached = [False] * self.NUM_AGENTS

        obs = [self._get_obs(i) for i in range(self.NUM_AGENTS)]
        return obs, {}

    # -------------------------------------------------------
    # STEP (CZYSTY PRIMAL)
    # -------------------------------------------------------
    def step(self, actions):
        assert len(actions) == self.NUM_AGENTS
        self.steps += 1

        rewards = [0.0] * self.NUM_AGENTS
        old_positions = list(self.agent_positions)

        # 1. Wylicz proponowane ruchy
        proposed = []
        for i, a in enumerate(actions):
            if self.agent_reached[i]:
                proposed.append(self.agent_positions[i])
                continue

            x, y = self.agent_positions[i]
            dx, dy = self.ACTIONS[a]
            nx, ny = x + dx, y + dy

            # poza planszą lub w przeszkodę -> stay
            if not (0 <= nx < self.GRID_SIZE and 0 <= ny < self.GRID_SIZE):
                nx, ny = x, y
            elif self.OBSTACLES[nx, ny] == 1:
                nx, ny = x, y

            proposed.append((nx, ny))

        # 2. Vertex collisions
        final_pos = list(self.agent_positions)
        counts = {}
        for p in proposed:
            counts[p] = counts.get(p, 0) + 1

        for i in range(self.NUM_AGENTS):
            if self.agent_reached[i]:
                continue
            if counts[proposed[i]] > 1:
                rewards[i] += self.R_COLLISION
                final_pos[i] = old_positions[i]
            else:
                final_pos[i] = proposed[i]

        # 3. Edge collisions
        for i in range(self.NUM_AGENTS):
            for j in range(i + 1, self.NUM_AGENTS):
                if self.agent_reached[i] and self.agent_reached[j]:
                    continue

                old_i, old_j = old_positions[i], old_positions[j]
                new_i, new_j = final_pos[i], final_pos[j]

                if new_i == old_j and new_j == old_i:
                    final_pos[i] = old_i
                    final_pos[j] = old_j
                    rewards[i] += self.R_COLLISION
                    rewards[j] += self.R_COLLISION

        # 4. Update positions
        self.agent_positions = final_pos

        # 5. Oblicz nagrody PRIMAL-style
        success = True
        for i in range(self.NUM_AGENTS):
            x, y = self.agent_positions[i]
            gx, gy = self.agent_goals[i]

            at_goal = (x, y) == (gx, gy)
            if at_goal:
                self.agent_reached[i] = True

            # Ruch lub stay
            if at_goal:
                rewards[i] += self.R_STAY_ON
            else:
                success = False
                if actions[i] == 0:      # stay off-goal
                    rewards[i] += self.R_STAY_OFF
                else:                    # ruch
                    rewards[i] += self.R_MOVE

        # 6. Zakończenie epizodu
        done = False
        if success:
            # WSZYSCY na celach → +20 do każdego agenta
            rewards = [r + self.R_SUCCESS for r in rewards]
            done = True

        if self.steps >= self.max_steps:
            done = True

        obs = [self._get_obs(i) for i in range(self.NUM_AGENTS)]
        info = {"num_agents_at_goal": sum(self.agent_reached)}

        return obs, rewards, done, info

    # -------------------------------------------------------
    # OBSERWACJE
    # -------------------------------------------------------
    def _get_obs(self, agent_id):
        fov = np.zeros((self.FOV_SIZE, self.FOV_SIZE, 4), dtype=np.float32)
        ax, ay = self.agent_positions[agent_id]
        half = self.FOV_SIZE // 2

        # wypełnij FOV
        for fx in range(self.FOV_SIZE):
            for fy in range(self.FOV_SIZE):
                gx = ax + (fx - half)
                gy = ay + (fy - half)

                if not (0 <= gx < self.GRID_SIZE and 0 <= gy < self.GRID_SIZE):
                    continue

                # przeszkody
                if self.OBSTACLES[gx, gy] == 1:
                    fov[fx, fy, 0] = 1.0

                # agenci
                for j, (px, py) in enumerate(self.agent_positions):
                    if j != agent_id and (px, py) == (gx, gy):
                        fov[fx, fy, 1] = 1.0

                # cele
                for (tx, ty) in self.agent_goals:
                    if (tx, ty) == (gx, gy):
                        fov[fx, fy, 2] = 1.0

        # własna pozycja
        fov[half, half, 3] = 1.0

        # goal vec
        gx, gy = self.agent_goals[agent_id]
        dx, dy = gx - ax, gy - ay
        dist = np.sqrt(dx*dx + dy*dy)
        if dist > 0:
            goal_vec = np.array([dx/dist, dy/dist, dist], dtype=np.float32)
        else:
            goal_vec = np.array([0, 0, 0], dtype=np.float32)

        # action mask
        action_mask = np.zeros(len(self.ACTIONS), dtype=np.float32)
        for i, (dx, dy) in enumerate(self.ACTIONS):
            nx, ny = ax + dx, ay + dy
            if (
                0 <= nx < self.GRID_SIZE
                and 0 <= ny < self.GRID_SIZE
                and self.OBSTACLES[nx, ny] == 0
            ):
                action_mask[i] = 1.0

        return {"fov": fov, "goal_vec": goal_vec, "action_mask": action_mask}
