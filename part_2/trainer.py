def train(env, agent, episodes, print_every=500):
    cum_rewards = []

    for i in range(episodes):
        reward_sum = 0

        observation = env.reset()
        action = agent.begin_episode(observation)
        done = False

        while not done:
            observation, reward, done = env.step(action)
            action = agent.act(observation, reward, done)
            reward_sum += reward

        cum_rewards.append(reward_sum)
        if i % print_every == 0:
            print(f"Episode {i}; Mean reward = {cum_rewards[-1]}")
    
    return cum_rewards
