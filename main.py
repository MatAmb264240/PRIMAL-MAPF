import csv
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
    n_envs = 8

    # Create the vectorized environment for training
    vec_env = make_vec_env(
        make_env(),
        n_envs=n_envs,
    )

    # Specify policy arguments
    policy_kwargs = dict(
        features_extractor_class=MAPFFeatureExtractor,
        features_extractor_kwargs=dict(
            num_agents=4,
            fov_size=10,
            n_channels=4,
        ),
        net_arch=dict(pi=[], vf=[]),  # Simple net architecture
    )

    try:
        # Try to load the pre-trained IL model
        model = RecurrentPPO.load("ppo_trained_agent", env=vec_env)
        print("Loaded ppo_trained_agent")
    except Exception as e:
        print(f"Error loading IL model: {e}")
        print("Creating a fresh RecurrentPPO model.")
        model = RecurrentPPO(
            policy="MlpLstmPolicy",
            env=vec_env,
            learning_rate=3e-4,
            gamma=0.99,
            n_steps=128,
            batch_size=256,
            n_epochs=4,
            gae_lambda=0.95,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            policy_kwargs=policy_kwargs,
            verbose=1,
        )

    # Train the model
    total_timesteps = 2_000_000
    model.learn(total_timesteps=total_timesteps)

    # Save the trained model
    model.save("ppo_trained_agent")

    # ====== TEST ======
    test_env = MAPF_SB3Env(
        grid_size=10,
        num_agents=4,
        fov_size=10,
        obstacle_density=0.2,
        max_steps=64,
    )

    # Prepare to write results to CSV
    with open('test_results_2.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Episode", "Test Reward", "Agents at Goal", "Elapsed Time"])

        episode = 1
        obs, info = test_env.reset()
        ep_reward = 0.0
        terminated = False
        truncated = False

        # Initialize LSTM states
        lstm_states = None
        episode_starts = True

        while not (terminated or truncated):
            # Predict the action and pass the LSTM states
            action, lstm_states = model.predict(
                obs,
                state=lstm_states,
                episode_start=episode_starts,
                deterministic=True,
            )
            obs, reward, terminated, truncated, info = test_env.step(action)
            ep_reward += float(reward)

            # After an episode ends, reset the LSTM state
            episode_starts = terminated or truncated

            # Log the results after each episode
            if terminated or truncated:
                agents_at_goal = info.get("num_agents_at_goal", 0)
                writer.writerow([episode, ep_reward, agents_at_goal, test_env._env.steps])
                print(f"Episode {episode}: Reward = {ep_reward}, Agents at Goal = {agents_at_goal}")
                episode += 1
                ep_reward = 0.0  # Reset reward for next episode
                obs, info = test_env.reset()  # Reset the environment for the next episode

        print("Test completed. Results saved to 'test_results.csv'.")


if __name__ == "__main__":
    main()
