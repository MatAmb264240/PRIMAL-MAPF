# mapf_env.py
import numpy as np
import random
import gymnasium as gym
from gymnasium import spaces
from collections import deque


class SimpleMAPFEnv(gym.Env):
    """
    Proste środowisko MAPF na siatce z nagrodami w stylu PRIMAL-lite.

    - wielu agentów na wspólnej planszy
    - każdy ma własny start i własny cel
    - lokalne FOV wokół agenta
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        grid_size: int = 8,
        num_agents: int = 3,
        fov_size: int = 10,
        obstacle_density: float = 0.1,
        max_steps: int = 64,
    ):
        super().__init__()
        self.GRID_SIZE = grid_size
        self.NUM_AGENTS = num_agents
        self.FOV_SIZE = fov_size
        self.OBSTACLE_DENSITY = obstacle_density
        self.max_steps = max_steps
        self.seed_value = None

        # akcje: 0 stay, 1 right, 2 left, 3 down, 4 up
        self.ACTIONS = [(0, 0), (0, 1), (0, -1), (1, 0), (-1, 0)]

        # Nagrody PRIMAL-lite:
        # - ruch w stronę celu ma być wyraźnie opłacalny
        # - stanie poza celem jest karane
        # - kolizje są mocno karane
        # - dojście do celu bardzo nagradzane
        self.REWARD_MOVE = -0.1        # ruch poza celem
        self.REWARD_STAY_OFF = -0.1    # stanie poza celem
        self.REWARD_STAY_ON = +0.1      # stanie na celu
        self.REWARD_BLOCKING = -2.0     # niewykorzystane na razie, ale zostawione
        self.REWARD_COLLISION = -2.0
        self.REWARD_OBSTACLE_COLLISION = -0.5
        self.REWARD_FINISH = 50.0      # pierwsze wejście na cel - świadome przegięcie pod RL
        self.SHAPING_FACTOR = 0.3       # wzmocnienie za zbliżanie się do celu

        self.action_space = spaces.Discrete(len(self.ACTIONS))
        self.observation_space = spaces.Dict(
            {
                "fov": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(self.FOV_SIZE, self.FOV_SIZE, 4),
                    dtype=np.float32,
                ),
                "goal_vec": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(3,),
                    dtype=np.float32,
                ),
                "action_mask": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(len(self.ACTIONS),),
                    dtype=np.float32,
                ),
            }
        )

        self.OBSTACLES = None
        self.agent_positions = None
        self.agent_goals = None
        self.agent_reached_goal = None
        self.steps = 0

    def seed(self, seed=None):
        self.seed_value = seed
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    def get_cbs_description(self):
        obstacles = []
        for x in range(self.GRID_SIZE):
            for y in range(self.GRID_SIZE):
                if self.OBSTACLES[x, y] == 1:
                    obstacles.append((int(x), int(y)))

        starts = [(int(x), int(y)) for (x, y) in self.agent_positions]
        goals = [(int(gx), int(gy)) for (gx, gy) in self.agent_goals]

        return {
            "grid_size": int(self.GRID_SIZE),
            "obstacles": obstacles,
            "starts": starts,
            "goals": goals,
        }

    def debug_print_map(self):
        print("\nInformacje o agentach i ich celach:")
        for i, (agent_pos, goal_pos) in enumerate(
            zip(self.agent_positions, self.agent_goals)
        ):
            ax, ay = agent_pos
            gx, gy = goal_pos
            print(f"Agent {i} - Start: ({ax}, {ay}), Cel: ({gx}, {gy})")

        print("\nPrzeszkody:")
        for x in range(self.GRID_SIZE):
            for y in range(self.GRID_SIZE):
                if self.OBSTACLES[x, y] == 1:
                    print(f"Obstacle ({x}, {y})")
        print()

    def reset(self):
        """
        Zwraca listę obserwacji, po jednej dla każdego agenta.
        """
        self.steps = 0

        # generowanie przeszkód
        self.OBSTACLES = np.zeros(
            (self.GRID_SIZE, self.GRID_SIZE), dtype=np.int32
        )
        for x in range(self.GRID_SIZE):
            for y in range(self.GRID_SIZE):
                if random.random() < self.OBSTACLE_DENSITY:
                    self.OBSTACLES[x, y] = 1

        # losowanie startów i celów, tak aby nie stały na przeszkodach i cele były unikalne
        self.agent_positions = []
        self.agent_goals = []
        self.agent_reached_goal = [False for _ in range(self.NUM_AGENTS)]
        occupied = set()

        for _ in range(self.NUM_AGENTS):
            # start
            while True:
                sx = random.randint(0, self.GRID_SIZE - 1)
                sy = random.randint(0, self.GRID_SIZE - 1)
                if self.OBSTACLES[sx, sy] == 0 and (sx, sy) not in occupied:
                    start = (sx, sy)
                    occupied.add(start)
                    break

            # goal
            while True:
                gx = random.randint(0, self.GRID_SIZE - 1)
                gy = random.randint(0, self.GRID_SIZE - 1)
                if (
                    self.OBSTACLES[gx, gy] == 0
                    and (gx, gy) != start
                    and (gx, gy) not in occupied
                ):
                    goal = (gx, gy)
                    occupied.add(goal)
                    break

            self.agent_positions.append(start)
            self.agent_goals.append(goal)

        observations = [self._get_obs(i) for i in range(self.NUM_AGENTS)]
        return observations

    def step(self, actions):
        """
        actions: lista intów długości NUM_AGENTS
        """
        assert len(actions) == self.NUM_AGENTS
        self.steps += 1

        rewards = [0.0 for _ in range(self.NUM_AGENTS)]
        old_positions = list(self.agent_positions)

        # 1. Proponowane nowe pozycje
        proposed = []
        for i, a in enumerate(actions):
            # jeśli agent już dotarł do celu, nie ruszamy go
            if self.agent_reached_goal[i]:
                proposed.append(self.agent_positions[i])
                continue

            x, y = self.agent_positions[i]
            dx, dy = self.ACTIONS[int(a)]
            nx, ny = x + dx, y + dy

            # wyjście poza planszę lub wejście w przeszkodę -> agent zostaje, ale bez collision penalty
            if not (0 <= nx < self.GRID_SIZE and 0 <= ny < self.GRID_SIZE):
                nx, ny = x, y
                rewards[i] += self.REWARD_OBSTACLE_COLLISION
            elif self.OBSTACLES[nx, ny] == 1:
                nx, ny = x, y
                rewards[i] += self.REWARD_OBSTACLE_COLLISION

            proposed.append((nx, ny))

        # 2. Vertex collisions
        final_positions = list(self.agent_positions)
        counts = {}
        for pos in proposed:
            counts[pos] = counts.get(pos, 0) + 1

        for i in range(self.NUM_AGENTS):
            if counts[proposed[i]] > 1 and not self.agent_reached_goal[i]:
                # kolizja tylko jeśli agent jeszcze nie jest "zamrożony"
                final_positions[i] = self.agent_positions[i]
                rewards[i] += self.REWARD_COLLISION
            else:
                final_positions[i] = proposed[i]

        # 3. Edge (swap) collisions
        for i in range(self.NUM_AGENTS):
            for j in range(i + 1, self.NUM_AGENTS):
                if self.agent_reached_goal[i] and self.agent_reached_goal[j]:
                    continue

                old_i = self.agent_positions[i]
                old_j = self.agent_positions[j]
                new_i = final_positions[i]
                new_j = final_positions[j]

                if new_i == old_j and new_j == old_i:
                    final_positions[i] = old_i
                    final_positions[j] = old_j
                    if not self.agent_reached_goal[i]:
                        rewards[i] += self.REWARD_COLLISION
                    if not self.agent_reached_goal[j]:
                        rewards[j] += self.REWARD_COLLISION

        # 4. Entering occupied
        for i in range(self.NUM_AGENTS):
            if self.agent_reached_goal[i]:
                continue

            for j in range(self.NUM_AGENTS):
                if i == j:
                    continue

                old_j = self.agent_positions[j]
                new_i = final_positions[i]
                new_j = final_positions[j]

                # i próbuje wejść w pozycję j, a j się nie ruszył
                if new_i == old_j and new_j == old_j:
                    final_positions[i] = self.agent_positions[i]
                    rewards[i] += self.REWARD_COLLISION

        # zaktualizuj pozycje
        self.agent_positions = final_positions

        # 5. Reward shaping + FINISH
        done = True
        for i in range(self.NUM_AGENTS):
            ox, oy = old_positions[i]
            x, y = self.agent_positions[i]
            gx, gy = self.agent_goals[i]
            a = int(actions[i])

            # jeśli agent był wcześniej na celu i został zamrożony
            if self.agent_reached_goal[i] and (ox, oy) == (gx, gy):
                # drobna nagroda za stanie na celu
                rewards[i] += self.REWARD_STAY_ON
                continue

            old_dist = abs(ox - gx) + abs(oy - gy)
            new_dist = abs(x - gx) + abs(y - gy)

            # heurystyczne zbliżanie się do celu
            rewards[i] += self.SHAPING_FACTOR * (old_dist - new_dist)

            at_goal = (x, y) == (gx, gy)

            if at_goal and not self.agent_reached_goal[i]:
                rewards[i] += self.REWARD_FINISH
                self.agent_reached_goal[i] = True

            if at_goal:
                # na celu - jeśli agent ruszył się z celu, lekkie ukaranie
                if a == 0:
                    rewards[i] += self.REWARD_STAY_ON
                else:
                    rewards[i] -= 1.0
            else:
                # poza celem
                if a == 0:
                    rewards[i] += self.REWARD_STAY_OFF
                else:
                    rewards[i] += self.REWARD_MOVE

            if not at_goal:
                done = False

        # 6. Warunki końca epizodu
        if sum(self.agent_reached_goal) == self.NUM_AGENTS:
            done = True
        if self.steps >= self.max_steps:
            done = True

        observations = [self._get_obs(i) for i in range(self.NUM_AGENTS)]
        info = {
            "num_agents_at_goal": sum(self.agent_reached_goal),
            "pct_agents_at_goal": sum(self.agent_reached_goal) / self.NUM_AGENTS,
        }
        return observations, rewards, done, info

    def _get_obs(self, agent_id: int):
        """
        Lokalny FOV wokół agenta + wektor do celu + maska akcji.
        FOV_SIZE może być mniejsze lub większe od GRID_SIZE.
        """
        fov = np.zeros((self.FOV_SIZE, self.FOV_SIZE, 4), dtype=np.float32)

        ax, ay = self.agent_positions[agent_id]
        half = self.FOV_SIZE // 2

        for fx in range(self.FOV_SIZE):
            for fy in range(self.FOV_SIZE):
                gx = ax + (fx - half)
                gy = ay + (fy - half)

                if not (0 <= gx < self.GRID_SIZE and 0 <= gy < self.GRID_SIZE):
                    continue

                # kanał 0: przeszkody
                if self.OBSTACLES[gx, gy] == 1:
                    fov[fx, fy, 0] = 1.0

                # kanał 1: inni agenci
                for i, (px, py) in enumerate(self.agent_positions):
                    if i != agent_id and (px, py) == (gx, gy):
                        fov[fx, fy, 1] = 1.0

                # kanał 2: cele agentów
                for (tx, ty) in self.agent_goals:
                    if (tx, ty) == (gx, gy):
                        fov[fx, fy, 2] = 1.0

        # kanał 3: własna pozycja - środek FOV
        fov[half, half, 3] = 1.0

        # goal_vec
        gx, gy = self.agent_goals[agent_id]
        dx = gx - ax
        dy = gy - ay
        dist = np.sqrt(dx * dx + dy * dy)
        if dist > 0:
            goal_vec = np.array([dx / dist, dy / dist, dist], dtype=np.float32)
        else:
            goal_vec = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        # action_mask
        action_mask = np.zeros(len(self.ACTIONS), dtype=np.float32)
        for a_idx, (dx, dy) in enumerate(self.ACTIONS):
            nx, ny = ax + dx, ay + dy
            if (
                0 <= nx < self.GRID_SIZE
                and 0 <= ny < self.GRID_SIZE
                and self.OBSTACLES[nx, ny] == 0
            ):
                action_mask[a_idx] = 1.0

        return {
            "fov": fov,
            "goal_vec": goal_vec,
            "action_mask": action_mask,
        }

    # BFS i blocking zostawiam, jakbyś kiedyś chciał blocking penalty jak w PRIMAL

    def _bfs_dist(self, start, goal, extra_blocked=None):
        if extra_blocked is None:
            extra_blocked = set()

        sx, sy = start
        gx, gy = goal

        if (sx, sy) == (gx, gy):
            return 0

        blocked = set(extra_blocked)
        for x in range(self.GRID_SIZE):
            for y in range(self.GRID_SIZE):
                if self.OBSTACLES[x, y] == 1:
                    blocked.add((x, y))

        if (sx, sy) in blocked or (gx, gy) in blocked:
            return None

        q = deque()
        q.append((sx, sy, 0))
        visited = set()
        visited.add((sx, sy))

        moves = [(1, 0), (-1, 0), (0, 1), (0, -1)]

        while q:
            x, y, d = q.popleft()
            for dx, dy in moves:
                nx, ny = x + dx, y + dy
                if not (0 <= nx < self.GRID_SIZE and 0 <= ny < self.GRID_SIZE):
                    continue
                if (nx, ny) in blocked:
                    continue
                if (nx, ny) in visited:
                    continue
                if (nx, ny) == (gx, gy):
                    return d + 1
                visited.add((nx, ny))
                q.append((nx, ny, d + 1))

        return None

    def _compute_blocking_labels(self, actions):
        labels = [0 for _ in range(self.NUM_AGENTS)]
        positions = list(self.agent_positions)
        all_agents_as_blocked = set(positions)

        for j in range(self.NUM_AGENTS):
            xj, yj = positions[j]
            gxj, gyj = self.agent_goals[j]
            at_goal_j = (xj, yj) == (gxj, gyj)
            staying = int(actions[j]) == 0

            if not (at_goal_j and staying):
                continue

            for i in range(self.NUM_AGENTS):
                if i == j:
                    continue
                si = positions[i]
                gi = self.agent_goals[i]

                if si == gi:
                    continue

                dist_with = self._bfs_dist(
                    si, gi, extra_blocked=all_agents_as_blocked
                )
                if dist_with is None:
                    continue

                blocked_without_j = set(all_agents_as_blocked)
                if positions[j] in blocked_without_j:
                    blocked_without_j.remove(positions[j])

                dist_without = self._bfs_dist(
                    si, gi, extra_blocked=blocked_without_j
                )
                if dist_without is None:
                    continue

                if dist_with - dist_without >= 10:
                    labels[j] = 1
                    break

        return labels
