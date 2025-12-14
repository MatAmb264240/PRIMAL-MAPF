# generate_expert_data.py
import numpy as np
from mapf_env_sb3 import MAPF_SB3Env
from solve_cbs import solve_cbs


def action_from_step(p_old, p_new, actions):
    dx = p_new[0] - p_old[0]
    dy = p_new[1] - p_old[1]
    for idx, (adx, ady) in enumerate(actions):
        if (dx, dy) == (adx, ady):
            return idx
    return 0  # WAIT


def main():
    grid_size = 10
    num_agents = 4
    fov_size = 10
    obstacle_density = 0.2
    max_steps = 64

    env = MAPF_SB3Env(
        grid_size=grid_size,
        num_agents=num_agents,
        fov_size=fov_size,
        obstacle_density=obstacle_density,
        max_steps=max_steps,
    )

    all_obs = []
    all_actions = []

    num_episodes = 3000

    for ep in range(num_episodes):
        obs, info = env.reset()
        desc = env._env.get_cbs_description()

        try:
            paths = solve_cbs(
                grid_size=desc["grid_size"],
                obstacles=desc["obstacles"],
                starts=desc["starts"],
                goals=desc["goals"],
            )
        except Exception as e:
            print(f"[EP {ep}] CBS failed: {e}")
            continue

        max_path_len = max(len(p) for p in paths.values())
        T = min(max_path_len - 1, max_steps)

        for t in range(T):
            # === KRYTYCZNE: deep copy ===
            all_obs.append(np.array(obs, copy=True))

            joint_actions = []
            for agent_id in range(num_agents):
                path = paths[agent_id]

                idx_old = min(t, len(path) - 1)
                idx_new = min(t + 1, len(path) - 1)

                p_old = path[idx_old]
                p_new = path[idx_new]

                a = action_from_step(p_old, p_new, env._env.ACTIONS)
                joint_actions.append(a)

            all_actions.append(joint_actions)

            obs, reward, terminated, truncated, info = env.step(joint_actions)
            if terminated or truncated:
                break

        print(f"[EP {ep}] collected {len(all_obs)} samples")

    all_obs = np.array(all_obs, dtype=np.float32)
    all_actions = np.array(all_actions, dtype=np.int64)

    print("Final dataset shapes:", all_obs.shape, all_actions.shape)
    np.savez("expert_dataset.npz", obs=all_obs, actions=all_actions)
    print("Saved to expert_dataset.npz")


if __name__ == "__main__":
    main()
