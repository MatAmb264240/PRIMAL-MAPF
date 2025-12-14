import os
import numpy as np
import torch as th
from torch.utils.data import DataLoader, TensorDataset

from sb3_contrib import RecurrentPPO
from stable_baselines3.common.env_util import make_vec_env

from mapf_env_sb3 import MAPF_SB3Env
from custom_policy import MAPFFeatureExtractor


# =========================================================
# ENV FACTORY (jak wcześniej)
# =========================================================

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


# =========================================================
# POLICY EVALUATION (jak w RL, ale bez gradientów)
# =========================================================

def eval_policy(policy, env, n_episodes=20):
    policy.eval()

    total_rew = 0.0
    total_len = 0
    success = 0

    for _ in range(n_episodes):
        obs, info = env.reset()
        done = False
        truncated = False

        ep_rew = 0.0
        ep_len = 0

        lstm_states = None
        episode_start = True

        while not (done or truncated):
            with th.no_grad():
                action, lstm_states = policy.predict(
                    obs,
                    state=lstm_states,
                    episode_start=episode_start,
                    deterministic=True,
                )

            obs, reward, done, truncated, info = env.step(action)
            ep_rew += sum(reward)
            ep_len += 1
            episode_start = done or truncated

        total_rew += ep_rew
        total_len += ep_len
        if info.get("num_agents_at_goal", 0) == env._env.NUM_AGENTS:
            success += 1

    policy.train()

    return (
        total_rew / n_episodes,
        total_len / n_episodes,
        success / n_episodes,
    )


# =========================================================
# MAIN
# =========================================================

def main():
    # -----------------------------------------------------
    # LOAD DATASET
    # -----------------------------------------------------
    print("Loading expert dataset...")
    data = np.load("expert_dataset.npz")

    obs = th.tensor(data["obs"], dtype=th.float32)
    actions = th.tensor(data["actions"], dtype=th.long)

    dataset = TensorDataset(obs, actions)
    loader = DataLoader(dataset, batch_size=256, shuffle=True)

    # -----------------------------------------------------
    # ENV (tylko do policy i ewaluacji)
    # -----------------------------------------------------
    vec_env = make_vec_env(make_env(), n_envs=1)

    policy_kwargs = dict(
        features_extractor_class=MAPFFeatureExtractor,
        features_extractor_kwargs=dict(
            num_agents=4,
            fov_size=10,
            n_channels=4,
        ),
        net_arch=dict(pi=[], vf=[]),
    )

    model_path = "ppo_trained_agent"

    # -----------------------------------------------------
    # LOAD MODEL (BEZ FALLBACKÓW)
    # -----------------------------------------------------
    model = RecurrentPPO.load(
        model_path,
        env=vec_env,
        device="auto",
    )
    print("MODEL LOADED FROM:", model_path)

    policy = model.policy
    device = policy.device
    print("DEVICE:", device)

    # -----------------------------------------------------
    # 🔒 FREEZE LSTM (KRYTYCZNE)
    # -----------------------------------------------------
    for p in policy.lstm_actor.parameters():
        p.requires_grad = False
    for p in policy.lstm_critic.parameters():
        p.requires_grad = False

    # -----------------------------------------------------
    # OPTIMIZER – tylko parametry trainowalne
    # -----------------------------------------------------
    optimizer = th.optim.Adam(
        filter(lambda p: p.requires_grad, policy.parameters()),
        lr=1e-5,
    )

    hidden_size = policy.lstm_actor.hidden_size

    num_epochs = 10
    best_reward = -1e9

    # -----------------------------------------------------
    # EWALUACJA PRZED IL
    # -----------------------------------------------------
    base_rew, base_len, base_succ = eval_policy(
        policy, vec_env.envs[0]
    )
    print(
        f"[BEFORE IL] rew={base_rew:.1f} | "
        f"len={base_len:.1f} | success={base_succ:.2f}"
    )

    # -----------------------------------------------------
    # IL LOOP
    # -----------------------------------------------------
    for epoch in range(num_epochs):
        total_loss = 0.0
        total_n = 0

        for batch_obs, batch_actions in loader:
            batch_obs = batch_obs.to(device)
            batch_actions = batch_actions.to(device)

            B = batch_obs.shape[0]

            lstm_states = (
                th.zeros((1, B, hidden_size), device=device),
                th.zeros((1, B, hidden_size), device=device),
            )
            episode_starts = th.ones((B,), dtype=th.float32, device=device)

            dist, _ = policy.get_distribution(
                batch_obs,
                lstm_states,
                episode_starts,
            )

            log_probs = dist.log_prob(batch_actions)
            loss = -log_probs.mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * B
            total_n += B

        avg_loss = total_loss / total_n

        # -------------------------------------------------
        # EWALUACJA PO EPOCE
        # -------------------------------------------------
        avg_rew, avg_len, success = eval_policy(
            policy, vec_env.envs[0]
        )

        print(
            f"[IL] Epoch {epoch+1}/{num_epochs} | "
            f"loss={avg_loss:.3f} | "
            f"rew={avg_rew:.1f} | "
            f"len={avg_len:.1f} | "
            f"success={success:.2f}"
        )

        # -------------------------------------------------
        # EARLY STOPPING
        # -------------------------------------------------
        if avg_rew > best_reward:
            best_reward = avg_rew
            model.save(model_path)
            print("💾 model saved (better reward)")
        else:
            print("⚠️ reward dropped → stopping IL")
            break

    print("IL finished.")


if __name__ == "__main__":
    main()
