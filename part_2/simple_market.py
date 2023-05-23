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
