import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from config import config # Importer la configuration du fichier config.py
import gymnasium as gym

# Créer l'environnement en utilisant la configuration fournie
env = gym.make("racetrack-v0", render_mode="rgb_array")
env.unwrapped.configure(config)

class PolicyNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(PolicyNetwork, self).__init__()
        self.dense1 = nn.Linear(state_size, 64)
        self.dense2 = nn.Linear(64, 64)
        self.output_layer = nn.Linear(64, action_size)

    def forward(self, state):
        x = torch.relu(self.dense1(state))
        x = torch.relu(self.dense2(x))
        action_probs = torch.softmax(self.output_layer(x), dim=-1)
        return action_probs

class PolicyAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.model = PolicyNetwork(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    def train(self, states, actions, advantages):
        self.optimizer.zero_grad()
        action_probs = self.model(states)
        chosen_actions = torch.nn.functional.one_hot(actions, self.action_size).float()
        log_probs = torch.log(torch.sum(action_probs * chosen_actions, dim=1))
        loss = -torch.mean(log_probs * advantages)
        loss.backward()
        self.optimizer.step()

    def get_action(self, state):
        with torch.no_grad():
            action_probs = self.model(state)
            action = action_probs.item()  # Sélectionner l'action avec la plus grande probabilité
        return action


def train_policy_gradient():
    state_size = env.observation_space.shape[0]  # Taille de l'espace des observations
    action_size = env.action_space.shape[0]  # Taille de l'espace des actions
    print("Observation space:", env.observation_space)
    print("Action space:", env.action_space)

    agent = PolicyAgent(state_size, action_size)

    # Entraînement
    episodes = 100
    batch_size = 32
    gamma = 0.99  # Facteur d'actualisation
    rewards = []

    for episode in range(episodes):
        episode_states = []
        episode_actions = []
        episode_rewards = []

        # Initialisation de l'environnement
        state = env.reset()
        total_reward = 0

        while True:
            # Extraire les éléments du tuple
            environment_grid, state_info = state

            # Convertir l'environnement en un tenseur PyTorch
            environment_tensor = torch.tensor(environment_grid, dtype=torch.float32)

            # Ajouter une dimension pour correspondre à la forme attendue par le modèle
            environment_tensor = environment_tensor.unsqueeze(0)  # Ajouter des dimensions pour le batch_size et les canaux

            # Convertir les informations d'état en un tenseur PyTorch
            state_info_tensor = torch.tensor([state_info['speed'], state_info['crashed'], state_info['action'], state_info['rewards']], dtype=torch.float32)

            # Concaténer les deux tenseurs
            state_tensor = torch.cat((environment_tensor, state_info_tensor), dim=1)

            # Choix de l'action
            action = agent.get_action(state_tensor)

            # Mise à jour de l'environnement et obtention de la récompense
            next_state1, reward, done, _, next_state2 = env.step(action)
            next_state = (next_state1, next_state2)

            # Enregistrer l'état, l'action et la récompense
            episode_states.append(state)
            episode_actions.append(action)
            episode_rewards.append(reward)

            # Actualisation de l'état et de la récompense totale
            state = next_state
            total_reward += reward

            # Condition de fin
            if done:  # Si l'épisode est terminé
                break

        # Calcul des avantages
        discounted_rewards = np.zeros_like(episode_rewards)
        cumulative_rewards = 0
        for t in reversed(range(len(episode_rewards))):
            cumulative_rewards = episode_rewards[t] + gamma * cumulative_rewards
            discounted_rewards[t] = cumulative_rewards
        mean_reward = np.mean(discounted_rewards)
        std_reward = np.std(discounted_rewards) if np.std(discounted_rewards) > 0 else 1
        advantages = (discounted_rewards - mean_reward) / std_reward

        # Entraînement de l'agent
        agent.train(torch.tensor(np.vstack(episode_states)).float(), torch.tensor(np.array(episode_actions)), torch.tensor(advantages))

        rewards.append(total_reward)
        if episode % 10 == 0:
            print("Episode:", episode, "Total Reward:", total_reward)

    # Sauvegarde des poids du modèle
    torch.save(agent.model.state_dict(), "policy_gradient_weights.pth")

if __name__ == "__main__":
    train_policy_gradient()
