from generators.generator import GENERATOR
import numpy as np

class SABR(GENERATOR):

    @staticmethod
    def generate(T=5, n=252, alpha=0.2, beta=1, rho=-0.3, vol_of_vol=0.2, initial_price=100):
        """
        Generate a synthetic stock time series using the SABR model.
        
        Parameters:
        T (float): Number of years
        N (int): Number of time steps per year
        alpha (float): Initial volatility
        beta (float): Beta parameter that controls the elasticity of variance (0 <= beta <= 1)
        rho (float): Correlation between the asset price and volatility (typically -1 to 1)
        vol_of_vol (float): Volatility of volatility
        initial_price (float): Starting stock price

        Returns:
        tuple: A tuple containing:
            - prices (numpy.ndarray): Simulated asset prices.
            - tt (numpy.ndarray): Time intervals corresponding to the asset prices, shape (n+1),.
        """
        dt = 1 / n
        prices = np.zeros((n*T) + 1)
        volatilities = np.zeros((n*T) + 1)
        
        # Initial conditions
        prices[0] = initial_price
        volatilities[0] = alpha
        
        # Generate correlated Brownian motions
        dW_price = np.random.normal(0, np.sqrt(dt), n*T)
        dW_vol = rho * dW_price + np.sqrt(1 - rho**2) * np.random.normal(0, np.sqrt(dt), n*T)
        
        # Simulate the time series
        for t in range(1, (n * T) + 1):
            volatilities[t] = max(volatilities[t - 1] + vol_of_vol * volatilities[t - 1] * dW_vol[t - 1], 0)  # Ensure non-negative volatility
            prices[t] = prices[t - 1] * np.exp(
                (volatilities[t - 1]**beta) * dW_price[t - 1]
            )
        
        timesteps = np.linspace(0, T, (n*T)+1)
        tt = np.full(shape=((n*T)+1), fill_value=timesteps)
        # return prices, np.linspace(0, T, n*T + 1)
        return prices, tt