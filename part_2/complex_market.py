import numpy as np


class MarketEnv:
    def __init__(self, sales_model, prime_cost):
        self.sales_model = sales_model
        self.prime_cost = prime_cost
        self.k = 0.001
    
    def reset(self):
        self.t = 1
        self.prev_price = None
        return self.t, self.prev_price
    
    def _get_sales(self, t, price):
        return self.sales_model.predict(self.t, price)
    
    def step(self, action):
        sales = self._get_sales(self.t, action)
        
        if self.t == 1:
            scale_parameter = 1
        else:
            scale_parameter = np.tanh(self.k * self._get_sales(self.t - 1, self.prev_price))
    
        profit = sales * scale_parameter * (action - self.prime_cost)
        self.t += 1
        self.prev_price = action
    
        return (self.t, self.prev_price), profit, self.t == 53


# It will be useful to use inheritance to implement agents for both types of the environment.
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
    
    def get_optimal_action(self, observation):
        # TODO: Your code here. Implement method that returns an optimal action.
        # This method will be used to compare performance with greedy agent.
        pass
