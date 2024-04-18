import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from copy import deepcopy
import gymnasium as gym
import time
import os
import matplotlib.pyplot as plt
from IPython.display import clear_output

from config import config

# Créer l'environnement en utilisant la configuration fournie
env = gym.make("racetrack-v0", render_mode="rgb_array")
env.unwrapped.configure(config)

# cartpole = gym.make("CartPole-v1", render_mode="rgb_array")

# Créer les agents
class RandomAgent:
    def __init__(self, observation_space, action_space):
        self.action_space = action_space

    def get_action(self, state, **kwargs):
        # Choose a random float between -1 and 1
        return self.action_space.sample()
    
    def update(self, state, action, reward, next_state, done):
        pass

def eval_agent(agent, env, n_episodes=10):
    env_copy = deepcopy(env)
    episode_rewards = np.zeros(n_episodes)
    for i in range (n_episodes):
        state, _ = env_copy.reset()
        done = False
        total_reward = 0
        while not done:
            action = agent.get_action(state)
            next_state, reward, terminated, _, info = env_copy.step(action)
            # next_state, reward, terminated, truncated, _ = env_copy.step(action)
            total_reward += reward
            state = next_state
            done = terminated or info['crashed'] or info['rewards']['collision_reward'] or not info['rewards']['on_road_reward']
            
        episode_rewards[i] = total_reward
    return episode_rewards

def run_one_episode(env, agent, display = True):
    display_env = deepcopy(env)
    state, _ = display_env.reset()
    done = False
    total_reward = 0
    while not done:
        action = agent.get_action(state)
        print(display_env.step(action))
        next_state, reward, terminated, _, info = display_env.step(action)
        # next_state, reward, done, _, _ = display_env.step(action)
        total_reward += reward
        state = next_state
        done = terminated or info['crashed'] or info['rewards']['collision_reward'] or not info['rewards']['on_road_reward']
        if display:
            clear_output(wait=True)
            plt.imshow(display_env.render())
            plt.show()
    if display:
        display_env.close()
    print(f"Total reward: {total_reward}")

# agent = RandomAgent(env.observation_space, env.action_space)
# run_one_episode(env, agent, display=True)

def train(env, agent, n_episodes, eval_every=10, reward_threshold = 300, n_eval = 10):
    total_time = 0
    for ep in range(n_episodes):
        done = False
        state, _ = env.reset()
        while not done:
            action = agent.get_action(state)
            next_state, reward, terminated, _, info = env.step(action)
            if not info['rewards']['on_road_reward']: # If the car is off the road
                reward -= 1
            reward += 0.01 # Reward for staying on the road
            agent.update(state, action, reward, done, next_state)
            state = next_state
            done = terminated or info['crashed'] or info['rewards']['collision_reward'] or not info['rewards']['on_road_reward']
            total_time += 1
            if ep == n_episodes -1 :
                clear_output(wait=True)
                plt.imshow(env.render())
                plt.show()
                env.close()

        if (ep+1) % eval_every == 0:
            mean_reward = np.mean(eval_agent(agent, env, n_eval))
            print(f"Episode {ep+1}, Mean reward: {mean_reward}")
            if mean_reward > reward_threshold:
                print(f"Solved in {ep} episodes!")
                break
    print("Finished training")
    # Save the model
    # torch.save(agent.policy_net.state_dict(), "reinforcement_batch.pth")
    return

def resize(x, n_action):
    # Resize the action to [-1, 1]
    return x*2/n_action -1

class Net(nn.Module):
    def __init__(self, obs_size, hidden_size, n_actions):
        super(Net, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            # nn.Linear(hidden_size, hidden_size*2),
            # nn.ReLU(),
            # nn.Linear(hidden_size*2, hidden_size),
            # nn.ReLU(),
            nn.Linear(hidden_size, n_actions)
        )
    
    def forward(self, x):
        return self.net(x)
    
class ReinforceSqueleton:
    def __init__(
        self,
        action_space,
        observation_space,
        gamma,
        episode_batch_size,
        learning_rate,
    ):
        self.action_space = action_space
        self.observation_space = observation_space
        self.gamma = gamma
        self.episode_batch_size = episode_batch_size
        self.learning_rate = learning_rate
        self.reset()
    
    def update(self, state, action, reward, done, next_state):
        pass

    def get_action(self, state):
        flat_state = torch.tensor(state.flatten()).float()
        with torch.no_grad():
            action_probs = self.policy_net.forward(flat_state)
            # Get the action with the highest probability
            action = action_probs.argmax()
            action = resize(action, 21)
            action = action.item() # output is a tensor of size 1
        return [action]
    
    def reset(self):

        # Observation_space = Box(-inf, inf, (2,12,12))
        # Action space = Box(-1, 1, (1,))

        obs_size = 288 # self.observation_space.shape = 2*12*12
        n_actions = 21 # self.action_space.shape
        hidden_size = 128
        
        self.policy_net = Net(obs_size, hidden_size, n_actions)
        self.current_episode = []
        self.scores = []
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        self.n_eps = 0


# agent = ReinforceSqueleton(env.action_space, env.observation_space, gamma=0.99, episode_batch_size=1, learning_rate=0.01)
# run_one_episode(env, agent, display=True)

class Reinforce(ReinforceSqueleton):

    def gradient_returns(self, rewards, gamma):
        """
        Turns a list of rewards into a list of rewards*gamma**t
        """
        G = 0
        returns_list = []
        T = len(rewards)
        full_gamma = np.power(gamma, T)
        for t in range(T):
            G = rewards[T-t-1] + G*gamma
            full_gamma = full_gamma/gamma
            returns_list.append(G*full_gamma)
        return torch.tensor(returns_list[::-1])
    
    def update(self, state, action, reward, done, next_state):

        self.current_episode.append((state, action, reward))
        if done:
            states, actions, rewards = zip(*self.current_episode)
            returns = self.gradient_returns(rewards, self.gamma)
            flat_states = torch.tensor(states.flatten()).float()
            returns = (returns - returns.mean())/ (returns.std() + 1e-9)
            action_probs = self.policy_net.forward(flat_states)
            action_probs = action_probs.item() # action_probs.gather(1, actions.view(-1, 1)).squeeze()
            loss = -torch.log(action_probs)*returns
            # loss = loss.mean()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.current_episode = []
        return

# agent = Reinforce(env.action_space, env.observation_space, gamma=0.95, episode_batch_size=1, learning_rate=0.01)
# start_time = time.time()
# train(env, agent, n_episodes=15000, eval_every=50, reward_threshold=200, n_eval=10)
# end_time = time.time()
# print(f"Training time: {(end_time - start_time)//60} min" )

class ReinforceBatch(Reinforce):
    
    # Update only after a batch of sequence of episodes
    def __init__(
            self,
            action_space,
            observation_space,
            gamma,
            episode_batch_size,
            learning_rate,
    ):
        super().__init__(action_space, observation_space, gamma, episode_batch_size, learning_rate)
        self.episode_buffer = []

    def update(self, state, action, reward, done, next_state):
        self.episode_buffer.append((state, action, reward))
        if done:
            self.n_eps += 1
            if self.n_eps % self.episode_batch_size == 0:
                self.learn_from_buffer()
                self.episode_buffer = []
    
    def learn_from_buffer(self):
        states, actions, rewards = zip(*self.episode_buffer)
        returns = self.gradient_returns(rewards, self.gamma)
        flat_states = torch.tensor(states.flatten()).float()
        returns = (returns - returns.mean())/ (returns.std() + 1e-9)
        action_probs = self.policy_net.forward(flat_states)
        action_probs = action_probs.item()
        action = action_probs.argmax()
        action = resize(action, 21)
        action = action.item()
        loss = -torch.log(action)*returns
        loss = loss.mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return
    
# agent = ReinforceBatch(env.action_space, env.observation_space, gamma=0.99, episode_batch_size=32, learning_rate=0.0005)
# start_time = time.time()
# train(env, agent, n_episodes=1000, eval_every=50, reward_threshold=200, n_eval=10)
# end_time = time.time()
# print(f"Training time: {(end_time - start_time)//60} min" )


class REINFORCEBatch(Reinforce):
    def update(self, state, action, reward, terminated, next_state):

        self.current_episode.append((
            torch.tensor(state).unsqueeze(0),
            torch.tensor([[action]], dtype=torch.int64),
            torch.tensor([reward]),
        )
        )

        if terminated:
            self.n_eps += 1

            states, actions, rewards = tuple(
                [torch.cat(data) for data in zip(*self.current_episode)]
            )

            current_episode_returns = self.gradient_returns(rewards, self.gamma)

            unn_log_probs = self.policy_net.forward(states)
            log_probs = unn_log_probs - torch.log(torch.sum(torch.exp(unn_log_probs), dim=1)).unsqueeze(1)
            self.scores.append(torch.dot(log_probs.gather(1, actions).squeeze(), current_episode_returns).unsqueeze(0))
            self.current_episode = []

            if (self.n_eps % self.episode_batch_size)==0:
                self.optimizer.zero_grad()
                full_neg_score = - torch.cat(self.scores).sum() / self.episode_batch_size
                full_neg_score.backward()
                self.optimizer.step()
                
                self.scores = []


agent = REINFORCEBatch(env.action_space, env.observation_space, gamma=0.95, episode_batch_size=32, learning_rate=0.001)
start_time = time.time()
train(env, agent, n_episodes=1500, eval_every=50, reward_threshold=200, n_eval=10)
end_time = time.time()
print(f"Training time: {(end_time - start_time)//60} min" )