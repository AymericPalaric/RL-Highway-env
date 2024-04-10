import gymnasium as gym

env = gym.make("racetrack-v0", render_mode="rgb_array")

config = {
            "observation": {
                "type": "Kinematics",
                "features": ['x', 'y', 'vx', 'vy', 'cos_h', 'sin_h'],
                "scales": [100, 100, 5, 5, 1, 1],
                "normalize": False
            },
            


            "action": {
                "type": "ContinuousAction",
                "longitudinal": False,
                "lateral": True,
                "target_speeds": [0, 5, 10]
            },
            "simulation_frequency": 15,
            "policy_frequency": 5,
            "duration": 150,
            "collision_reward": -5,
            "lane_centering_cost": 10,
            "controlled_vehicles": 1,
            "other_vehicles": 1,
            "screen_width": 600,
            "screen_height": 600,
            "centering_position": [0.5, 0.5],
        }

env.unwrapped.configure(config)
print(env.reset())