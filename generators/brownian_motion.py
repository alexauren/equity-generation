import numpy as np
import matplotlib.pyplot as plt
from generators.generator import GENERATOR

class BROWNIAN_MOTION(GENERATOR):

    @staticmethod
    def generate(mu=0.1, n=252, T=5, M=1, S0=100, sigma=0.25):     
        """
        Simulates asset prices using the Geometric Brownian Motion model.

        Parameters:
        mu (float): The drift coefficient, representing the expected return.
        n (int): Number of time steps per year.
        T (float): Number of years.
        M (int): The number of simulation paths.
        S0 (float): The initial asset price.
        sigma (float): The volatility coefficient.

        Returns:
        tuple: A tuple containing:
            - St (numpy.ndarray): Simulated asset prices, shape (n+1, M).
            - tt (numpy.ndarray): Time intervals corresponding to the asset prices, shape (n+1, M).
        """

        dt = 1 /n
        
        St = np.exp(
            (mu - sigma ** 2 / 2) * dt
            + sigma * np.random.normal(0, np.sqrt(dt), size=(M,n * T)).T
        )
        St = np.vstack([np.ones(M), St])
        St = S0 * St.cumprod(axis=0)

        time = np.linspace(0,T,(n*T)+1)

        # Require numpy array that is the same shape as St
        tt = np.full(shape=(M,(n*T)+1), fill_value=time).T
        return (St, tt)