import tensorflow as tf

from model import ACNet
from agent import A3CAgent
from trainer import A3CTrainer
from mapf_env import SimpleMAPFEnv

def main():
    num_actions = 5

    model = ACNet(num_actions=num_actions)
    dummy_fov = tf.zeros((1, 10, 10, 4), dtype=tf.float32)
    dummy_goal = tf.zeros((1, 3), dtype=tf.float32)
    model(dummy_fov, dummy_goal, state=None, training=False)

    optimizer = tf.keras.optimizers.Nadam(learning_rate=3e-4)
    agent = A3CAgent(model=model, optimizer=optimizer, gamma=0.95)

    trainer = A3CTrainer(
        agent=agent,
        env_class=SimpleMAPFEnv,
        grid_size=10,
        num_agents=4,
        obstacle_density=0.2,
        num_episodes=10000,
        max_steps=64,
        use_curriculum=True,
        use_il=True,
        il_fraction=0.5,
    )

    trainer.train()


if __name__ == "__main__":
    main()
