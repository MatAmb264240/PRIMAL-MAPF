import tensorflow as tf
from mapf_env import SimpleMAPFEnv
from buffer import ExperienceBuffer
from cbs_wrapper import solve_cbs
from agent import A3CAgent
import random
import math

class A3CTrainer:
    def __init__(
        self,
        agent: A3CAgent,
        env_class,
        grid_size: int = 16,
        num_agents: int = 4,
        obstacle_density: float = 0.1,
        num_episodes: int = 1500,
        max_steps: int = 256,
        use_curriculum: bool = True,
        use_il: bool = False,
        il_fraction: float = 0.5,
    ):
        self.agent = agent
        self.env_class = env_class
        self.grid_size = grid_size
        self.num_agents = num_agents
        self.obstacle_density = obstacle_density
        self.num_episodes = num_episodes
        self.max_steps = max_steps
        self.use_il = use_il
        self.il_fraction = il_fraction
        self.use_curriculum = use_curriculum

        # wyłącz GPU jeśli jest
        try:
            tf.config.set_visible_devices([], "GPU")
        except Exception:
            pass

    def _get_obstacle_density_for_episode(self, episode_idx: int):
        """
        Curriculum zbliżone do PRIMAL:
        jeśli use_curriculum=True, próbkujemy gęstość przeszkód
        z trójkątnego rozkładu w [0, base_obstacle_density],
        z pikiem w 0.8 * base_obstacle_density.
        """
        if not self.use_curriculum:
            return self.obstacle_density

        low = 0.0
        high = self.obstacle_density
        peak = 0.8 * high

        u = random.random()
        c = (peak - low) / (high - low + 1e-8)
        if u < c:
            density = low + math.sqrt(u * (high - low) * (peak - low))
        else:
            density = high - math.sqrt((1 - u) * (high - low) * (high - peak))

        return float(max(0.0, min(high, density)))

    def _merge_buffers(self, buffers):
        merged = ExperienceBuffer(gamma=self.agent.gamma)
        for buf in buffers:
            merged.states.extend(buf.states)
            merged.actions.extend(buf.actions)
            merged.rewards.extend(buf.rewards)
            merged.values.extend(buf.values)
            merged.dones.extend(buf.dones)
        return merged

    def _validate_cbs_paths(self, env, cbs_paths, episode_idx):
        """
        Sprawdza czy CBS zwrócił sensowne ścieżki:
        - dla każdego agenta jest path
        - path zaczyna się na aktualnym starcie
        - path kończy się na goalu
        - path ma długość co najmniej 2
        Zwraca True/False.
        """
        if not isinstance(cbs_paths, dict):
            print(f"[WARN] CBS returned non dict in episode {episode_idx}: {type(cbs_paths)}")
            return False

        for agent_id in range(env.NUM_AGENTS):
            path = cbs_paths.get(agent_id)
            if not path or len(path) < 2:
                print(f"[WARN] CBS missing or too short path for agent {agent_id} in episode {episode_idx}.")
                return False

            start_env = tuple(env.agent_positions[agent_id])
            goal_env = tuple(env.agent_goals[agent_id])
            start_path = tuple(path[0])
            goal_path = tuple(path[-1])

            if start_env != start_path:
                print(
                    f"[WARN] CBS path start mismatch for agent {agent_id} in episode {episode_idx}: "
                    f"path[0]={start_path} vs env_start={start_env}"
                )
                return False

            if goal_env != goal_path:
                print(
                    f"[WARN] CBS path goal mismatch for agent {agent_id} in episode {episode_idx}: "
                    f"path[-1]={goal_path} vs env_goal={goal_env}"
                )
                return False

        return True

    def train(self):
        for episode in range(self.num_episodes):
            obstacle_density = self._get_obstacle_density_for_episode(episode)

            env = self.env_class(
                grid_size=self.grid_size,
                num_agents=self.num_agents,
                obstacle_density=obstacle_density,
                max_steps=self.max_steps,
            )

            observations = env.reset()
            done = False
            episode_reward = 0.0
            step_idx = 0

            # wyczyść bufor na początku epizodu
            self.agent.buffer.clear()

            # decide if this episode uses IL (imitacja eksperta CBS)
            il_this_episode = False
            if self.use_il:
                if episode < 100:
                    il_this_episode = True
                elif random.random() < self.il_fraction:
                    il_this_episode = True

            buffers = [ExperienceBuffer(gamma=self.agent.gamma) for _ in range(self.num_agents)]
            lstm_states = [None for _ in range(self.num_agents)]

            info = {}

            cbs_paths = None
            if il_this_episode:
                # przygotuj ścieżki eksperta z CBS dla aktualnej mapy
                desc = env.get_cbs_description()
                try:
                    cbs_paths = solve_cbs(
                        desc["grid_size"],
                        desc["obstacles"],
                        desc["starts"],
                        desc["goals"],
                    )
                    env.debug_print_map()
                    print("\nCBS Paths:")
                    for agent_id, path in cbs_paths.items():
                        print(f"Agent {agent_id}: {path}")
                except Exception as e:
                    print(f"[WARN] CBS solve failed in episode {episode}: {e}. Falling back to RL.")
                    il_this_episode = False

                # walidacja ścieżek
                if il_this_episode:
                    if not self._validate_cbs_paths(env, cbs_paths, episode):
                        print(f"[WARN] Invalid CBS solution in episode {episode}, falling back to RL.")
                        il_this_episode = False

            if il_this_episode and cbs_paths is not None:
                max_path_len = max(len(p) for p in cbs_paths.values())
                max_il_steps = min(max_path_len - 1, self.max_steps)

                for step_idx in range(max_il_steps):
                    actions = []

                    # oblicz akcje eksperta dla każdego agenta
                    for agent_id in range(self.num_agents):
                        path = cbs_paths.get(agent_id)
                        if path is None or step_idx + 1 >= len(path):
                            a = 0  # stay
                        else:
                            x, y = path[step_idx]
                            nx, ny = path[step_idx + 1]
                            dx, dy = nx - x, ny - y

                            a = 0
                            for idx, (ax, ay) in enumerate(env.ACTIONS):
                                if ax == dx and ay == dy:
                                    a = idx
                                    break

                        actions.append(a)

                    next_observations, rewards, done, info = env.step(actions)

                    # zapisujemy (obs, akcja_eksperta, prawdziwy reward, value z sieci)
                    for agent_id in range(self.num_agents):
                        obs = observations[agent_id]
                        v = float(
                            self.agent.model.get_value(
                                obs["fov"],
                                obs["goal_vec"],
                            )
                        )
                        buffers[agent_id].add(
                            obs,
                            actions[agent_id],
                            float(rewards[agent_id]),
                            v,
                            done,
                        )

                    observations = next_observations
                    episode_reward += float(sum(rewards))

                    if done:
                        break

                # merge bufforów agentów
                merged_buffer = self._merge_buffers(buffers)
                self.agent.buffer = merged_buffer
                self.agent.update_from_buffer(il_mode=True)

            else:
                # klasyczne RL A3C
                while not done:
                    actions = []
                    values = []

                    for agent_id in range(self.num_agents):
                        obs = observations[agent_id]
                        action, value, new_state = self.agent.choose_action(
                            obs, lstm_state=lstm_states[agent_id]
                        )
                        actions.append(action)
                        values.append(value)
                        lstm_states[agent_id] = new_state

                    next_observations, rewards, done, info = env.step(actions)

                    for agent_id in range(self.num_agents):
                        obs = observations[agent_id]
                        buffers[agent_id].add(
                            obs, actions[agent_id], rewards[agent_id], values[agent_id], done
                        )

                    observations = next_observations
                    step_idx += 1
                    episode_reward += float(sum(rewards))

                # merge bufforów agentów
                merged_buffer = self._merge_buffers(buffers)
                self.agent.buffer = merged_buffer
                self.agent.update_from_buffer()

            num_done = info.get("num_agents_at_goal", 0)
            pct_done = info.get("pct_agents_at_goal", 0.0)
            avg_step_reward = episode_reward / max(1, step_idx)

            # Wypisanie podsumowania epizodu
            print(
                f"Epizod {episode + 1}/{self.num_episodes} | "
                f"reward: {episode_reward:.2f} | "
                f"kroki: {step_idx} | "
                f"avg r/step: {avg_step_reward:.3f} | "
                f"na celach: {num_done}/{self.num_agents} ({pct_done*100:.1f}%) | "
                f"obst_density: {obstacle_density:.2f} | "
                f"IL: {il_this_episode}"
            )

            # Dodaj logowanie dla agentów, którzy nie osiągnęli celu
            for i in range(self.num_agents):
                if not env.agent_reached_goal[i]:
                    print(f"Agent {i} nie osiągnął celu: {env.agent_goals[i]} {env.agent_positions[i]}")