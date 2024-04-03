import gymnasium as gym

env = gym.make("highway-fast-v0", render_mode="rgb_array")

config = {
    "observation": {
        "type": "OccupancyGrid",
        "vehicles_count": 10,
        "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
        "features_range": {
            "x": [-100, 100],
            "y": [-100, 100],
            "vx": [-20, 20],
            "vy": [-20, 20],
        "align_to_vehicle_axes": True
    },
    },
    "action": {
        "type": "ContinuousAction",
    },
    "lanes_count": 4,
    "vehicles_count": 10,
    "duration": 30,  # [s]
    "initial_spacing": 2,
    "collision_reward": -1,  # The reward received when colliding with a vehicle.
    "reward_speed_range": [20, 30],  # [m/s] The reward for high speed is mapped linearly from this range to [0, HighwayEnv.HIGH_SPEED_REWARD].
    "simulation_frequency": 15,  # [Hz]
    "policy_frequency": 1,  # [Hz]
    "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
    "screen_width": 600,  # [px]
    "screen_height": 150,  # [px]
    "centering_position": [0.3, 0.5],
    "scaling": 5.5,
    "on_road_reward":3,  # The reward received when on road.
    "show_trajectories": False,
    "render_agent": True,
    "offscreen_rendering": False
}


env.unwrapped.configure(config)
print(env.reset())