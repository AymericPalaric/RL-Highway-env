import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.optim.lr_scheduler import StepLR
from copy import deepcopy
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
import random
from utils import Memory
from nets import Net

DEVICE = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cuda" if torch.cuda.is_available() else "cpu")



def select_action(model, env, state, eps):
    state = torch.Tensor(state).to(DEVICE)
    # print(state.shape)
    with torch.no_grad():
        values = model(state.unsqueeze(0))

    # select a random action wih probability eps
    if random.random() <= eps:
        action = np.random.randint(0, env.action_space.n)
    else:
        action = np.argmax(values.cpu().squeeze().numpy())

    return action


def train(batch_size, current, target, optim, memory, gamma):

    states, actions, next_states, rewards, is_done = memory.sample(batch_size)

    q_values = current(states)

    next_q_values = current(next_states)
    next_q_state_values = target(next_states)

    q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
    next_q_value = next_q_state_values.gather(1, torch.max(next_q_values, 1)[1].unsqueeze(1)).squeeze(1)
    expected_q_value = rewards + gamma * next_q_value * (1 - is_done)

    loss = (q_value - expected_q_value.detach()).pow(2).mean()

    optim.zero_grad()
    loss.backward()
    optim.step()
    return loss.item()


def evaluate(Qmodel, env, repeats):
    """
    Runs a greedy policy with respect to the current Q-Network for "repeats" many episodes. Returns the average
    episode reward.
    """
    Qmodel.eval()
    perform = 0
    for _ in range(repeats):
        state = env.reset()[0]
        done = False
        while not done:
            state = torch.Tensor(state).to(DEVICE)
            with torch.no_grad():
                values = Qmodel(state.unsqueeze(0))
            action = np.argmax(values.cpu().squeeze().numpy())
            state, reward, done, truncated, info = env.step(action)
            done = done or truncated
            reward, done = correct_reward(done, env, info, reward)
            perform += reward
    Qmodel.train()
    return perform/repeats


def update_parameters(current_model, target_model):
    target_model.load_state_dict(current_model.state_dict())

def correct_reward(done, env, info, reward):
    """
    Update the reward to penalize going backwards and off the road.
    state observation: [presence, x, y, vx, vy, cos_h, sin_h] for each cell in a radius of 4 cells around the agent.
    """
    # get agent position, heading and speed
    x = env.vehicle.position[0]
    y = env.vehicle.position[1]
    speed = info["speed"]
    on_road = info["rewards"]["on_road_reward"]>0
    collision = info["rewards"]["collision_reward"]>0
    heading = env.vehicle.heading
    # print(heading)
    if not on_road:
        reward -= 2
        done = True
    if speed <= 20:
        reward -= 2
    if collision:
        reward -= 2
    # check the heading
    if (heading>np.pi/2 and heading<3*np.pi/2) or (heading<-np.pi/2 and heading>-3*np.pi/2):
        reward -= 1
    return reward, done


def main(env, gamma=0.99, lr=1e-3, min_episodes=20, eps=1, eps_decay=0.995, eps_min=0.01, update_step=10, batch_size=64, update_repeats=50,
         num_episodes=3000, seed=42, max_memory_size=50000, lr_gamma=0.9, lr_step=100, measure_step=100,
         measure_repeats=100, hidden_dim=64, horizon=np.inf, render=True, render_step=50):
    """
    :param gamma: reward discount factor
    :param lr: learning rate for the Q-Network
    :param min_episodes: we wait "min_episodes" many episodes in order to aggregate enough data before starting to train
    :param eps: probability to take a random action during training
    :param eps_decay: after every episode "eps" is multiplied by "eps_decay" to reduces exploration over time
    :param eps_min: minimal value of "eps"
    :param update_step: after "update_step" many episodes the Q-Network is trained "update_repeats" many times with a
    batch of size "batch_size" from the memory.
    :param batch_size: see above
    :param update_repeats: see above
    :param num_episodes: the number of episodes played in total
    :param seed: random seed for reproducibility
    :param max_memory_size: size of the replay memory
    :param lr_gamma: learning rate decay for the Q-Network
    :param lr_step: every "lr_step" episodes we decay the learning rate
    :param measure_step: every "measure_step" episode the performance is measured
    :param measure_repeats: the amount of episodes played in to asses performance
    :param hidden_dim: hidden dimensions for the Q_network
    :param env_name: name of the gym environment
    :param cnn: set to "True" when using environments with image observations like "Pong-v0"
    :param horizon: number of steps taken in the environment before terminating the episode (prevents very long episodes)
    :param render: if "True" renders the environment every "render_step" episodes
    :param render_step: see above
    :return: the trained Q-Network and the measured performances
    """

    Q_1 = Net(0, hidden_dim, env.action_space.n).to(DEVICE)
    Q_2 = Net(0, hidden_dim, env.action_space.n).to(DEVICE)
    
    # transfer parameters from Q_1 to Q_2
    update_parameters(Q_1, Q_2)

    # we only train Q_1
    for param in Q_2.parameters():
        param.requires_grad = False

    optimizer = optim.Adam(Q_1.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=lr_step, gamma=lr_gamma)

    memory = Memory(max_memory_size)
    performance = []
    ep_rewards = []
    loss = 0
    max_reward = -np.inf
    pbar = trange(num_episodes, postfix={"reward": 0, "lr": lr, "eps": eps, "loss": 0})
    for episode in pbar:
        # display the performance
        if (episode % measure_step == 0) and episode >= min_episodes:
            performance.append([episode, evaluate(Q_1, env, measure_repeats)])
            # torch.save(Q_1.state_dict(), "Q.pth")
            print("Episode: ", episode)
            print("rewards: ", performance[-1][1])
            print("lr: ", scheduler.get_lr()[0])
            print("eps: ", eps)

        state = env.reset()[0]
        memory.state.append(state)

        done = False
        i = 0
        ep_reward = 0
        full_reward = 0
        while not done:
            i += 1
            action = select_action(Q_2, env, state, eps)
            state, reward, done, truncated, info = env.step(action)
            done = done or truncated
            reward, done = correct_reward(done, env, info, reward)
            full_reward += reward

            if i > horizon:
                done = True

            # render the environment if render == True
            if render and episode % render_step == 0:
                env.render()

            # save state, action, reward sequence
            memory.update(state, action, reward, done)

        if episode >= min_episodes and episode % update_step == 0:
            for _ in range(update_repeats):
                loss = train(batch_size, Q_1, Q_2, optimizer, memory, gamma)

            # transfer new parameter from Q_1 to Q_2
            update_parameters(Q_1, Q_2)

        # update learning rate and eps
        scheduler.step()
        eps = max(eps*eps_decay, eps_min)
        ep_reward = full_reward
        if ep_reward > max_reward:
            max_reward = ep_reward
            # save the best model
            torch.save(Q_1.state_dict(), "best_Q.pth")
        ep_rewards.append(ep_reward)
        pbar.set_postfix(reward=ep_reward, lr=scheduler.get_lr()[0], eps=eps, loss=loss)

    return Q_1, ep_rewards

def run_episode(env, Q):
    state = env.reset()[0]
    done = False
    ep_reward = 0
    while not done:
        state = torch.Tensor(state).to(DEVICE)
        with torch.no_grad():
            values = Q(state.unsqueeze(0))
        action = np.argmax(values.cpu().squeeze().numpy())
        state, reward, done, truncated, info = env.step(action)
        done = done or truncated
        reward, done = correct_reward(done, env, info, reward)
        ep_reward += reward
        env.render()
    print("Reward: ", ep_reward)
    env.close()


if __name__=="__main__":
    from config import config, env

    # Q = Net(0, 128, env.action_space.n).to(DEVICE)
    # Q.load_state_dict(torch.load("final_Q.pth"))
    # run_episode(env, Q)

    Q, rewards = main(env, render=True, render_step=100, measure_step=1000, num_episodes=3_000, update_step=50, hidden_dim=128, lr=5e-3, measure_repeats=10, eps_decay=0.997)
    # # save the trained Q-Network
    torch.save(Q.state_dict(), "final_Q.pth")

    plt.plot(rewards)
    plt.xlabel("Episode")
    plt.ylabel("Performance")
    plt.show()
