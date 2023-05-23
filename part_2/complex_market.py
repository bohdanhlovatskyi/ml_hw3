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

