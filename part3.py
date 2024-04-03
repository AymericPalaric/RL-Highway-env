import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from copy import deepcopy

from config3 import env

class FlattenObservation(gym.Wrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)

    def observation(self, observation: np.ndarray) -> np.ndarray:
        return observation.flatten()
    


if __name__ == "__main__":
    train = True
    if train:
        n_cpu = 6
        batch_size = 256
        env_train = FlattenObservation(deepcopy(env))
        model = PPO(
            "MlpPolicy",
            env_train,
            policy_kwargs=dict(net_arch=[dict(pi=[256, 256], vf=[256, 256])]),
            n_steps=batch_size * 12 // n_cpu,
            batch_size=batch_size,
            n_epochs=10,
            learning_rate=1e-2,
            gamma=0.8,
            verbose=2,
        )
        # Train the agent
        model.learn(total_timesteps=int(10_000))
        # Save the agent
        model.save("highway_ppo/model")

    model = PPO.load("highway_ppo/model")

    env_display = FlattenObservation(deepcopy(env))

    for _ in range(10):
        obs, info = env_display.reset()
        done = truncated = False
        while not (done or truncated):
            action, _ = model.predict(obs)
            obs, reward, done, truncated, info = env_display.step(action)
            env_display.render()
