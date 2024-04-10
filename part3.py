import warnings
warnings.filterwarnings("ignore", message="WARN: env.compute_reward to get variables from other wrappers is deprecated and will be removed in v1.0, to get this variable you can do `env.unwrapped.compute_reward` for environment variables or `env.get_wrapper_attr('compute_reward')` that will search the reminding wrappers.")


import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO, A2C,SAC, DDPG
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3 import HerReplayBuffer, SAC

from copy import deepcopy
import torch
from config3 import env

class FlattenObservation(gym.Wrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)

    def observation(self, observation: np.ndarray) -> np.ndarray:
        return observation.flatten()
    


if __name__ == "__main__":
    train = False
    if train:
        env_train = FlattenObservation(deepcopy(env))
        # n_actions = env.action_space.shape[-1]
        # action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

        # model = DDPG("MultiInputPolicy", env, action_noise=action_noise, verbose=1)
        her_kwargs = dict(n_sampled_goal=4, goal_selection_strategy='future')
        model = SAC('MultiInputPolicy', env_train, replay_buffer_class=HerReplayBuffer,
                    replay_buffer_kwargs=her_kwargs, verbose=1,
                    tensorboard_log="logs",
                    buffer_size=int(1e6),
                    learning_rate=1e-3,
                    gamma=0.95, batch_size=1024, tau=0.05,
                    learning_starts=1000,  # Ajoutez cette ligne
                    device="cpu",
                    policy_kwargs=dict(net_arch=[512, 512, 512]))
        # model = A2C(
        #     "MultiInputPolicy",
        #     env_train,
        #     policy_kwargs=dict(net_arch=[dict(pi=[256, 256], vf=[256, 256])]),
        #     n_steps=batch_size * 12 // n_cpu,  # Ce paramètre détermine également la taille du buffer de trajectoires
        #     batch_size=batch_size,
        #     n_epochs=10,
        #     learning_rate=1e-3,
        #     gamma=0.9,
        #     verbose=1,
        #     
        # )

        # Train the agent
        model.learn(total_timesteps=int(30_000))
        # Save the agent
        model.save("highway_ppo/model")

    # model = PPO.load("highway_ppo/model")
    env_eval = FlattenObservation(deepcopy(env))

    model = SAC.load("highway_ppo/model", env=env_eval)

    env_display = FlattenObservation(deepcopy(env))

    for _ in range(10):
        obs, info = env_display.reset()
        done = truncated = False
        while not (done or truncated):
            action, _ = model.predict(obs)
            obs, reward, done, truncated, info = env_display.step(action)
            env_display.render()
