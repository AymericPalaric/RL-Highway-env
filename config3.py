import gymnasium as gym
import numpy as np

env = gym.make("parking-v0", render_mode="rgb_array")

config = {
            "observation": {
                "type": "KinematicsGoal",
                "features": ['x', 'y', 'vx', 'vy', 'cos_h', 'sin_h'],
                "scales": [100, 100, 5, 5, 1, 1],
                "normalize": False
            },
            "action": {
                "type": "ContinuousAction"
            },
            "reward_weights": [1, 0.3, 0, 0, 0.02, 0.02],
            "success_goal_reward": 0.12,
            "collision_reward": -5,
            "steering_range": np.deg2rad(90),
            "simulation_frequency": 40,
            "policy_frequency": 5,
            "duration": 40,
            "screen_width": 600,
            "screen_height": 300,
            "centering_position": [0.5, 0.5],
            "scaling": 7,
            "controlled_vehicles": 1,
            "vehicles_count": 3,
            "add_walls": True
            }

env.unwrapped.configure(config)
print(env.reset())