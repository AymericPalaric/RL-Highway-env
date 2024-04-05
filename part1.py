import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import random
from torch import optim
from copy import deepcopy
import matplotlib.pyplot as plt
from tqdm import tqdm, trange

DEVICE = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Net(nn.Module):
    def __init__(self, obs_size, hidden_size, n_actions):
        super(Net, self).__init__()
        # self.net = nn.Sequential(
        #     nn.Linear(obs_size, hidden_size),
        #     nn.ReLU(),
        #     nn.Linear(hidden_size, n_actions),
        # )
        self.conv1 = nn.Conv2d(7, 16, kernel_size=3, stride=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1)
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(32*4*4, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions),
        )

    def forward(self, x):
        # flatten x
        # x = x.view(x.size(0), -1)
        # x = x.unsqueeze(1)
        # print("x shape", x.shape)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.flatten(x)
        # print("x shape", x.shape)
        x = self.fc(x)

        return x


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, state, action, reward, terminated, next_state):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = (state, action, reward, terminated, next_state)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.choices(self.memory, k=batch_size)

    def __len__(self):
        return len(self.memory)

class DQN:
    def __init__(
        self,
        action_space,
        observation_space,
        gamma,
        batch_size,
        buffer_capacity,
        update_target_every,
        epsilon_start,
        decrease_epsilon_factor,
        epsilon_min,
        learning_rate,
    ):
        self.action_space = action_space
        self.observation_space = observation_space
        self.gamma = gamma

        self.batch_size = batch_size
        self.buffer_capacity = buffer_capacity
        self.update_target_every = update_target_every

        self.epsilon_start = epsilon_start
        self.decrease_epsilon_factor = (
            decrease_epsilon_factor  # larger -> more exploration
        )
        self.epsilon_min = epsilon_min

        self.learning_rate = learning_rate

        self.reset()

    def reset(self):
        hidden_size = 128

        obs_size = np.prod(self.observation_space.shape)
        # print("obs_size", obs_size)
        n_actions = self.action_space.n

        self.buffer = ReplayBuffer(self.buffer_capacity)
        self.q_net = Net(obs_size, hidden_size, n_actions)
        self.q_net.to(DEVICE)
        self.target_net = Net(obs_size, hidden_size, n_actions)
        self.target_net.to(DEVICE)

        self.loss_function = nn.MSELoss()
        self.optimizer = optim.Adam(
            params=self.q_net.parameters(), lr=self.learning_rate
        )

        self.epsilon = self.epsilon_start
        self.n_steps = 0
        self.n_eps = 0

    def update(self, state, action, reward, terminated, next_state):
        """
        ** TO BE COMPLETED **
        """
        # print("action in update", action)
        # add data to replay buffer
        self.buffer.push(torch.tensor(state).unsqueeze(0), 
                           torch.tensor([[action]], dtype=torch.int64), 
                           torch.tensor([reward]), 
                           torch.tensor([terminated], dtype=torch.int64), 
                           torch.tensor(next_state).unsqueeze(0),
                          )

        if len(self.buffer) < self.batch_size:
            return np.inf

        # get batch
        transitions = self.buffer.sample(self.batch_size)

        # Compute loss - TO BE IMPLEMENTED!
        # Hint: use the gather method from torch.

        # Compute the Q values for the current state
        (
            states_batch,
            actions_batch,
            rewards_batch,
            terminated_batch,
            next_states_batch
        ) = tuple(map(torch.cat, zip(*transitions)))
        states_batch = states_batch.to(DEVICE)
        actions_batch = actions_batch.to(DEVICE)
        q_values = self.q_net.forward(states_batch).gather(1, actions_batch)

        # compute the ideal Q values
        self.target_net.eval()
        with torch.no_grad():
            next_states_batch = next_states_batch.to(DEVICE)
            terminated_batch = terminated_batch.to(DEVICE)
            next_state_values = (1 - terminated_batch) * self.target_net(
                next_states_batch
            ).max(1)[0]
            next_state_values = next_state_values.to(DEVICE)
            rewards_batch = rewards_batch.to(DEVICE)
            targets = next_state_values * self.gamma + rewards_batch
            targets = targets.float()
        self.target_net.train()
        # print(targets.unsqueeze(1).shape, q_values.shape)
        # print("targets", targets.dtype)
        # print("q_values", q_values.dtype)
        loss = self.loss_function(q_values, targets.unsqueeze(1))
        # print("loss", loss)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if not ((self.n_steps + 1) % self.update_target_every):
            self.target_net.load_state_dict(self.q_net.state_dict())

        self.decrease_epsilon()

        self.n_steps += 1
        if terminated:
            self.n_eps += 1

        return loss.detach().cpu().numpy()

    def get_action(self, state, epsilon=None):
        """
        Return action according to an epsilon-greedy exploration policy
        """
        if epsilon is None:
            epsilon = self.epsilon

        if np.random.rand() < epsilon:
            action =  env.action_space.sample()
        else:
            action = np.argmax(self.get_q(state))
        # print("action in get action", action)
        return action

    def decrease_epsilon(self):
        self.epsilon = self.epsilon_min + (self.epsilon_start - self.epsilon_min) * (
            np.exp(-1.0 * self.n_eps / self.decrease_epsilon_factor)
        )
    
    def get_q(self, state):
        """
        Compute Q function for a states
        """
        # state_tensor = torch.tensor([state]).unsqueeze(0)
        # print("state in get_q", state)
        state_tensor = torch.tensor(state).unsqueeze(0)
        self.q_net.eval()
        with torch.no_grad():
            state_tensor = state_tensor.to(DEVICE)
            output = self.q_net.forward(state_tensor) # shape (1,  n_actions)
        # print("output", output.numpy()[0])
        self.q_net.train()
        return output.cpu().numpy()[0]


def eval_agent(agent, env, n_sim=5, verbose=False):
    """
    ** TO BE IMPLEMENTED **
    
    Monte Carlo evaluation of DQN agent.

    Repeat n_sim times:
        * Run the DQN policy until the environment reaches a terminal state (= one episode)
        * Compute the sum of rewards in this episode
        * Store the sum of rewards in the episode_rewards array.
    """
    env_copy = deepcopy(env)
    episode_rewards = np.zeros(n_sim)
    for i in trange(n_sim, disable=not verbose):
        state,_ = env_copy.reset()
        done = False
        while not done:
            action = agent.get_action(state)
            # print(env_copy.step(action))
            next_state, reward, terminated, truncated, _ = env_copy.step(action)
            state = next_state
            done = terminated or truncated
            episode_rewards[i] += reward
    return episode_rewards

def train(env, agent, N_episodes, eval_every=10, reward_threshold=300):
    total_time = 0
    state, _ = env.reset()
    losses = []
    agg_rewards = []
    # display the progress with the reward and the loss
    pbar = tqdm(range(N_episodes), postfix={"reward": 0.0, "loss": 0.0})
    for ep in pbar:
        done = False
        ep_reward = 0
        state, _ = env.reset()
        while not done: 
            action = agent.get_action(state)

            next_state, reward, terminated, truncated, _ = env.step(action)
            loss_val = agent.update(state, action, reward, terminated, next_state)
            ep_reward += reward

            pbar.set_postfix(
                {
                    "reward": ep_reward,
                    "loss": loss_val,
                    "epsilon": agent.epsilon,
                }
            )
            state = next_state
            losses.append(loss_val)

            done = terminated or truncated
            total_time += 1
        agg_rewards.append(ep_reward)
        if ((ep+1)% eval_every == 0):
            rewards = eval_agent(agent, env)
            print("episode =", ep+1, ", reward = ", np.mean(rewards))
            if np.mean(rewards) >= reward_threshold:
                break
                
    return losses, agg_rewards


def run_agent(agent, env):
    state, _ = env.reset()
    done = False
    full_reward = 0
    while not done:
        action = agent.get_action(state)
        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        env.render()
        full_reward += reward
    print("final reward", full_reward)
    env.close()

if __name__=="__main__":
    TRAIN = False
    from config import env, config

    
    if TRAIN:
        
        # print("config", env.config)
        env.reset()
        print("obs space:", env.observation_space)
        print("action space:", env.action_space, env.action_space.n)
        # print("action space sample:", env.action_space.sample())
        print("obs space sample:", env.observation_space.sample().shape)
        # print("step:", env.step(env.action_space.sample()))
        env.reset()

        action_space = env.action_space
        observation_space = env.observation_space
        gamma = 0.95
        batch_size = 128
        buffer_capacity = 10_000
        update_target_every = 256

        epsilon_start = 0.15
        decrease_epsilon_factor = 1000
        epsilon_min = 0.05

        learning_rate = 1e-3

        arguments = (action_space,
                    observation_space,
                    gamma,
                    batch_size,
                    buffer_capacity,
                    update_target_every, 
                    epsilon_start, 
                    decrease_epsilon_factor, 
                    epsilon_min,
                    learning_rate,
                )
        
        
        agent = DQN(*arguments)

        N_episodes = 1_000
        eval_every = 100
        reward_threshold = 300

        state, _ = env.reset()
        done = False
        full_reward = 0
        while not done:
            action = agent.get_action(state)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            # env.render()
            full_reward += reward
        print("Reward before training", full_reward)
        # env.close()

        env.reset()

        losses, agg_rewards = train(env, agent, N_episodes, eval_every=eval_every, reward_threshold=reward_threshold)
        
        plt.plot(losses)
        plt.title("Loss")
        plt.show()
        plt.savefig("loss.png")

        plt.figure()
        plt.plot(agg_rewards)
        plt.title("Reward")
        plt.show()
        plt.savefig("reward.png")

        rewards = eval_agent(agent, env, 5, verbose=True)
        print("")
        print("mean reward after training = ", np.mean(rewards))

        # save model
        torch.save(agent.q_net.state_dict(), "dqn_model.pth")
    # test run
    env = gym.make("highway-v0", render_mode="rgb_array")
    config["duration"] = 150
    env.unwrapped.configure(config)
    env.reset()
    print("obs space:", env.observation_space)
    print("action space:", env.action_space, env.action_space.n)
    action_space = env.action_space
    observation_space = env.observation_space
    gamma = 0.9
    batch_size = 64
    buffer_capacity = 10_000
    update_target_every = 128

    epsilon_start = 0.15
    decrease_epsilon_factor = 1000
    epsilon_min = 0.05

    learning_rate = 5e-3

    arguments = (action_space,
                observation_space,
                gamma,
                batch_size,
                buffer_capacity,
                update_target_every, 
                epsilon_start, 
                decrease_epsilon_factor, 
                epsilon_min,
                learning_rate,
            )
    state, _ = env.reset()
    agent_test = DQN(*arguments)
    if TRAIN:
        agent_test.q_net.load_state_dict(agent.q_net.state_dict())
    else:
        print("loading model")
        agent_test.q_net.load_state_dict(torch.load("dqn_model.pth"))
    # agent_test.target_net.load_state_dict(agent.target_net.state_dict())
    agent_test.epsilon = 0.0
    agent_test.q_net.eval()
    # agent_test.target_net.eval()
    
    run_agent(agent_test, env)

    env.close()