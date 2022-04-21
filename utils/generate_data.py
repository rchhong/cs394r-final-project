import torch

# For Actor-Critic
def generate_a2c_data(num_transitions, gamma, agent, net, env, device):
    games_data = []

    states = []
    actions = []
    rewards = []
    next_states = []

    curr_state = env.reset()
    done = False

    # Statistics
    # num_wins = 0
    # num_losses = 0
    # avg_reward_per_game = 0

    for _ in range(num_transitions):
        action = agent(torch.Tensor([curr_state['state']], device=device))[1][0]
        next_state, reward, done, _ = env.step(action)

        states.append(curr_state['state'])
        actions.append(action)
        rewards.append(reward)
        next_states.append(next_state['state'])

        curr_state = next_state

        if done:
            _, bootstrapped_value = net(torch.Tensor([curr_state['state']], device = device)).detach().squeeze().numpy()
            returns = compute_returns(bootstrapped_value, rewards, gamma)
            game_data = list(zip(states, actions, next_states, returns[::-1]))

            games_data.append([transition for transition in game_data])

            curr_state = env.reset()
            done = False

    # Finish off the stragglers
    if(len(states)):
        _, bootstrapped_value = net(torch.Tensor([curr_state['state']], device = device)).detach().squeeze().numpy()
        returns = compute_returns(bootstrapped_value, rewards, gamma)
        game_data = list(zip(states, actions, next_states, returns[::-1]))

        games_data.append([transition for transition in game_data])

    return games_data



def compute_returns(bootstrapped_value, rewards, gamma):
    returns = []

    ret = bootstrapped_value
    requires_bootstrap = False
    for reward in rewards[::-1]:
        if requires_bootstrap:
            ret = reward + gamma * ret
            returns.append(ret)
        else:
            returns.append(reward)
        requires_bootstrap = True
