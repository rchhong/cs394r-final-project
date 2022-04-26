import torch
import numpy as np

# For Actor-Critic
def generate_data(gamma, agent, net, env, device):
    game_data = []

    states = []
    actions = []
    rewards = []
    next_states = []

    curr_state = env.reset()
    done = False

    while not done:
        action, action_taken_logprob, state_value = agent(torch.Tensor([curr_state['state']], device=device))
        env_action = pred[1][0]
        net_action = pred[0][0]

        next_state, reward, done, _ = env.step(env_action)

        states.append(curr_state['state'])
        actions.append(net_action)
        rewards.append(reward)
        next_states.append(next_state['state'])

        curr_state = next_state

    # _, bootstrapped_value = net(torch.Tensor([curr_state['state']], device = device))
    # bootstrapped_value = bootstrapped_value.detach().squeeze().numpy()

    # MC
    bootstrapped_value = 0
    returns = compute_returns(bootstrapped_value, rewards, gamma)
    game_data = list(zip(states, actions, next_states, returns))

    # Finish off the stragglers
    assert(len(game_data) == 6)
    return game_data,



def compute_returns(rewards, gamma):
    returns = []

    g = 0
    for reward in rewards[::-1]:
        g = reward + gamma * g
        returns.append(g)


    return returns[::-1]
