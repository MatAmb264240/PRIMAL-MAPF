import os
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.env_util import make_vec_env
from mapf_env_sb3 import MAPF_SB3Env
import master_git.visualization


def make_env():
    def _init():
        return MAPF_SB3Env(
            grid_size=10,
            num_agents=4,
            fov_size=10,
            obstacle_density=0.1,
            max_steps=64,
        )
    return _init


def main():
    N_EPISODES = 5

    vec_env = make_vec_env(make_env(), n_envs=1)

    model = RecurrentPPO.load(
        "ppo_trained_agent",
        env=vec_env,
        device="cuda",
    )

    print("Loaded model.")

    for ep in range(1, N_EPISODES + 1):
        obs = vec_env.reset()

        lstm_states = None
        episode_starts = [True]
        done = False

        # --- WYCIĄGNIĘCIE PRAWDZIWEJ MAPY ZE ŚRODOWISKA ---
        env = vec_env.envs[0].env._env          # SB3 wrapper
        primal_env = env              # SimpleMAPFEnv

        obstacles = primal_env.OBSTACLES.copy()
        starts = primal_env.agent_positions.copy()
        goals = primal_env.agent_goals.copy()

        num_agents = primal_env.NUM_AGENTS

        # --- TRAJECTORIES ---
        trajectories = [[] for _ in range(num_agents)]
        for i in range(num_agents):
            trajectories[i].append(primal_env.agent_positions[i])

        # --- LOGGING ---
        episode_reward = 0.0
        episode_steps = 0

        # --- SYMULACJA EPIZODU ---
        while not done:
            action, lstm_states = model.predict(
                obs,
                state=lstm_states,
                episode_start=episode_starts,
                deterministic=False,
            )

            obs, reward, done, info = vec_env.step(action)

            # reward = suma rewardów agentów
            episode_reward += float(reward)
            episode_steps += 1

            # zbieranie pozycji do trajektorii
            for i in range(num_agents):
                trajectories[i].append(primal_env.agent_positions[i])

            episode_starts = [done]

        agents_at_goal = info[0].get("num_agents_at_goal", 0)

        # --- PRINT PODSUMOWANIA EPIZODU ---
        print(f"\n===== EPISODE {ep:03d} SUMMARY =====")
        print(f"Reward: {episode_reward:.2f}")
        print(f"Steps : {episode_steps}")
        print(f"Agents at goal: {agents_at_goal}/{num_agents}")
        print("Solution (trajectory lengths):")
        for i, traj in enumerate(trajectories):
            print(f"  Agent {i}: {len(traj)} steps")

        # --- ZAPIS ANIMACJI ---
        save_path = f"animations/episode_{ep:03d}.mp4"
        master_git.visualization.visualize_episode(
            obstacles, starts, goals, trajectories, save_path
        )
        print(f"Saved animation: {save_path}")


if __name__ == "__main__":
    main()