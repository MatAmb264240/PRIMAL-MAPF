from sb3_contrib import RecurrentPPO
from stable_baselines3.common.env_util import make_vec_env

from mapf_env_sb3 import MAPF_SB3Env
from custom_policy import MAPFFeatureExtractor


def make_env():
    def _init():
        return MAPF_SB3Env(
            grid_size=10,
            num_agents=4,
            fov_size=10,
            obstacle_density=0.2,
            max_steps=64,
        )
    return _init


def main():
    N_EPISODES = 100

    # VecEnv (SB3) – zgodnie z twoim treningiem
    vec_env = make_vec_env(make_env(), n_envs=1)

    model = RecurrentPPO.load(
        "ppo_trained_agent",
        env=vec_env,
        device="cuda",
    )

    print("Loaded ppo_trained_agent")

    success_count = 0
    total_reward = 0.0
    total_steps = 0

    for episode in range(1, N_EPISODES + 1):
        # VecEnv.reset() -> tylko obs
        obs = vec_env.reset()

        lstm_states = None
        episode_starts = [True]

        ep_reward = 0.0
        done = False
        steps = 0
        agents_at_goal = 0

        while not done:
            action, lstm_states = model.predict(
                obs,
                state=lstm_states,
                episode_start=episode_starts,
                deterministic=False,
            )

            # UWAGA: VecEnv.step() zwraca 4 wartości, NIE 5
            obs, reward, done, info = vec_env.step(action)

            ep_reward += float(reward)
            steps += 1

            episode_starts = [done]

            if done:
                agents_at_goal = info[0].get("num_agents_at_goal", 0)

        total_reward += ep_reward
        total_steps += steps
        if agents_at_goal == 4:
            success_count += 1

        print(
            f"Episode {episode:03d}: Reward = {ep_reward:.2f}, Steps = {steps}, Agents at Goal = {agents_at_goal}"
        )

    print("\n===== TEST SUMMARY (100 EPISODES) =====")
    print(f"Mean reward   : {total_reward / N_EPISODES:.2f}")
    print(f"Mean steps    : {total_steps / N_EPISODES:.1f}")
    print(f"Success rate  : {100.0 * success_count / N_EPISODES:.1f}%")


if __name__ == "__main__":
    main()