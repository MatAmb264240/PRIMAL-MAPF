# il_pretrain.py
import numpy as np
import torch as th
from torch.utils.data import DataLoader, TensorDataset

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
    print("Loading expert dataset...")
    data = np.load("expert_dataset.npz")
    obs = th.tensor(data["obs"], dtype=th.float32)
    actions = th.tensor(data["actions"], dtype=th.long)
    dataset = TensorDataset(obs, actions)
    loader = DataLoader(dataset, batch_size=256, shuffle=True)

    # init env just for policy
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

    model = RecurrentPPO(
        policy="MlpLstmPolicy",
        env=vec_env,
        learning_rate=3e-4,
        n_steps=128,
        batch_size=256,
        n_epochs=4,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=policy_kwargs,
        verbose=1,
    )

    policy = model.policy
    device = policy.device
    optimizer = th.optim.Adam(policy.parameters(), lr=1e-4)

    # 🔥 poprawne LSTM size
    hidden_size = policy.lstm_actor.hidden_size

    num_epochs = 15

    for epoch in range(num_epochs):
        total_loss = 0
        total_n = 0

        for batch_obs, batch_actions in loader:
            batch_obs = batch_obs.to(device)
            batch_actions = batch_actions.to(device)

            B = batch_obs.shape[0]

            # 🔥 poprawna inicjalizacja LSTM
            lstm_states = (
                th.zeros((1, B, hidden_size), device=device),  # h
                th.zeros((1, B, hidden_size), device=device),  # c
            )
            episode_starts = th.ones((B,), dtype=th.float32, device=device)

            # 🔥 RecurrentPPO distribution call
            dist, _ = policy.get_distribution(
                batch_obs,
                lstm_states,
                episode_starts
            )

            log_probs = dist.log_prob(batch_actions)
            loss = -log_probs.mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * B
            total_n += B

        print(f"[IL] Epoch {epoch+1}/{num_epochs} - loss = {total_loss/total_n:.4f}")

    model.save("ppo_il_pretrained")
    print("Saved IL model as ppo_il_pretrained.zip")


if __name__ == "__main__":
    main()