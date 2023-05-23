import datetime
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose


class SalesPredictor:
    def __init__(self, period, residuals_model):
        self.period = period
        self.residuals_model = residuals_model
    
    def fit(self, prices, sales):
        # TODO: implement a training procedure. "prices" has a datetime index.
        # The method should extract & save seasonal component and train estimator on residuals.
        # Note that a first week number should be saved in order to determine seasonality in the future.
        pass
    
        decomposition = seasonal_decompose(prices, model='additive', period=self.period)
        print(decomposition)
        decomposition.plot()
    
    def _predict_array(self, week, prices):
        # TODO: Implement prediction procedure. Use extracted seasonality and model trained on residuals.
        pass
    
    def predict(self, week, prices):
        if not isinstance(week, np.ndarray) and not isinstance(prices, np.ndarray):
            sales_pred = self._predict_array(np.array([week]), np.array([[prices]]))
            return sales_pred[0]
        return self._predict_array(week, prices)

if __name__ == "__main__":
    import os
    import pandas as pd
    
    from sklearn.linear_model import Lasso
    
    data_path = os.path.join("data", "sales_data.csv")
    df = pd.read_csv(data_path, index_col=0)
    print(df.head())
    
    sp = SalesPredictor(period=52, residuals_model=Lasso())
    sp.fit(df["Price"], df["Sales"])
    
    