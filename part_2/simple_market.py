import numpy as np


class MarketEnv:
    def __init__(self, sales_model, prime_cost):
        self.sales_model = sales_model
        self.prime_cost = prime_cost
    
    def reset(self):
        self.t = 1
        return self.t
    
    def step(self, action):
        sales = self.sales_model.predict(self.t, action)
        profit = sales * (action - self.prime_cost)
        self.t += 1

        return self.t, profit, self.t == 53


# Since tabular Q-learning deals with finite state-space, the agent should discretize continious observations.
class MarketAgent:
    def __init__(self, min_price, max_price, bins_number, learning_rate=0.1, discount_factor=0.96,
                 exploration_rate=0.98, exploration_decay_rate=0.99):
        # TODO: Your code here
        pass
    
    def begin_episode(self, observation):
        # TODO: Your code here. Implement method that returns an epsilon-greedy action.
        pass
    
    def act(self, observation, reward, done):
        # TODO: Your code here. Implement method that returns an epsilon-greedy action and updates Q-table.
        pass
