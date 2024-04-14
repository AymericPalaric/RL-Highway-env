import numpy as np
import torch
from config import Config
from train import PolicyNetwork

def inference():
    config = Config()
    state_size = 2 * config.num_lanes + 1  # Taille de l'état : position de la voiture principale + positions des autres voitures
    action_size = 3  # Gauche, droite, rien

    model = PolicyNetwork(state_size, action_size)
    model.load_state_dict(torch.load("policy_gradient_weights.pth"))
    model.eval()

    # Simulation de l'environnement
    state = np.random.random((1, state_size))  # État initial aléatoire
    done = False
    total_reward = 0

    while not done:
        # Choix de l'action
        with torch.no_grad():
            action_probs = model(torch.tensor(state).float())
            action = np.random.choice(action_size, p=action_probs.numpy().flatten())

        # Mise à jour de l'environnement et obtention de la récompense
        next_state = np.random.random((1, state_size))  # Prochain état aléatoire
        reward = np.random.random()  # Récompense aléatoire

        # Actualisation de l'état et de la récompense totale
        state = next_state
        total_reward += reward

        # Condition de fin
        if total_reward > 100:  # Condition arbitraire de fin de simulation
            done = True

    print("Total Reward:", total_reward)

if __name__ == "__main__":
    inference()
