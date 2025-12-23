print("-------")
import torch
print(torch.cuda.is_available())
import numpy as np

from sb3_contrib import RecurrentPPO
from stable_baselines3.common.env_util import make_vec_env

from mapf_env_sb3 import MAPF_SB3Env
from custom_policy import MAPFFeatureExtractor
from il_callback import OnlineILCallback

device = torch.device("cuda")

# =========================================================
# === LOAD EXPERT DATA ONCE
# =========================================================
expert_data = np.load("expert_dataset.npz")
EXPERT_OBS = expert_data["obs"]
EXPERT_ACTIONS = expert_data["actions"]


def load_one_expert_batch(batch_size=16):
    idx = np.random.choice(len(EXPERT_OBS), size=batch_size, replace=False)

    obs = torch.as_tensor(EXPERT_OBS[idx], dtype=torch.float32, device=device)
    actions = torch.as_tensor(EXPERT_ACTIONS[idx], dtype=torch.long, device=device)

    return obs, actions


# =========================================================
# === ENV FACTORY (CURRICULUM)
# =========================================================
def make_env(density):
    def _init():
        return MAPF_SB3Env(
            grid_size=10,
            num_agents=4,
            fov_size=10,
            obstacle_density=density,
            max_steps=64,
        )
    return _init


# =========================================================
# === MAIN
# =========================================================
def main():

    # -----------------------------------------------------
    # POLICY CONFIG
    # -----------------------------------------------------
    policy_kwargs = dict(
        features_extractor_class=MAPFFeatureExtractor,
        features_extractor_kwargs=dict(
            num_agents=4,
            fov_size=10,
            n_channels=4,
        ),
        net_arch=dict(pi=[], vf=[]),
    )

    # -----------------------------------------------------
    # INIT ENV (start density = 0.0)
    # -----------------------------------------------------
    current_density = 0.0
    vec_env = make_vec_env(make_env(current_density), n_envs=1)

    # -----------------------------------------------------
    # LOAD OR CREATE MODEL
    # -----------------------------------------------------
    try:
        model = RecurrentPPO.load(
            "ppo_trained_agent",
            env=vec_env,
            device=device,
        )
        print("Loaded ppo_trained_agent")
    except Exception:
        print("Creating new RecurrentPPO model")
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
            device=device,
        )

    # -----------------------------------------------------
    # IL CALLBACK
    # -----------------------------------------------------
    il_callback = OnlineILCallback(
        expert_loader_fn=load_one_expert_batch,
        il_coef=0.05,
        every_n_steps=1_000,
        verbose=2,
    )

    # -----------------------------------------------------
    # CURRICULUM TRAINING LOOP
    # -----------------------------------------------------
    TOTAL_STEPS = 1_000_000
    CURRICULUM_END = 0.2
    CHUNK = 5000

    steps_done = 0

    while steps_done < TOTAL_STEPS:
        progress = steps_done / TOTAL_STEPS
        new_density = CURRICULUM_END * progress

        vec_env.env_method("init_env", density=new_density)

        model.learn(
            total_timesteps=CHUNK,
            callback=il_callback
        )

        steps_done += CHUNK
        print(
            f"[CURRICULUM] steps={steps_done:,} "
            f"density={new_density:.3f}"
        )

    model.save("ppo_trained_agent")

    vec_env.close()
    print("Training finished.")


if __name__ == "__main__":
    main()
