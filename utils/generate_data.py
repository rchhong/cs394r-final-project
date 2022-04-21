import torch

# For Actor-Critic
def generate_a2c_data(batch_size, gamma, agent, net, env, device):
    # Need this for now to ensure accurate statistics reporting
    assert(batch_size % 6 == 0)
    games_data = []

    # num_wins = 0
    # num_losses = 0
    # avg_reward_per_game = 0

    while(len(games_data) < batch_size):

        states = []
        actions = []
        rewards = []
        next_states = []
        # Generate episodes
        curr_state = env.reset()
        done = False

        while not done:
            action = agent(torch.Tensor([curr_state['state']], device=device))[1][0]
            next_state, reward, done, _ = env.step(action)

            states.append(curr_state['state'])
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state['state'])

            curr_state = next_state

        returns = []
        _, bootstrapped_value = net(torch.Tensor([curr_state['state']], device = device))

        ret = bootstrapped_value.detach().squeeze().numpy()
        requires_bootstrap = False
        for reward in rewards[::-1]:
            if requires_bootstrap:
                ret = reward + gamma * ret
                returns.append(ret)
            else:
                returns.append(reward)
            requires_bootstrap = True
        game_data = list(zip(states, actions, next_states, returns[::-1]))
        games_data.append([transition for transition in game_data])

    return games_data[:batch_size]