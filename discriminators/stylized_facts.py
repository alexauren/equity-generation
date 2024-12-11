import numpy as np
import scipy.stats as stats
from visualization.plotter import PLOTTER
import matplotlib.pyplot as plt
from statsmodels.stats.diagnostic import acorr_ljungbox

class STYLIZED_FACTS:

    def __init__(self, data):
        self.data = data
    
    def change_data(self, data):
        self.data = data

    def check_fat_tail(self, significance_level: float = 0.05) -> bool:
        """
        Checks if a stock price time series follows the fat tail phenomenon.

        Parameters:
        - data (np.ndarray): 1D array of stock prices.
        - significance_level (float): The significance level for the test (default is 0.05).

        Returns:
        - bool: True if the series follows the fat tail phenomenon, False otherwise.
        """
        data = self.data
        log_returns = np.diff(np.log(data))
        
        # Fit the data to a normal distribution
        _, p_value = stats.kstest(log_returns, 'norm', args=(np.mean(log_returns), np.std(log_returns)))

        if p_value < significance_level:
            return True 
        else:
            return False 
        
    def check_volatility_clustering(self) -> bool: # can be improved
        """
        Checks if a stock price time series exhibits volatility clustering.

        Parameters:
        - data (np.ndarray): 1D array of stock prices.
        - window_size (int): The number of periods for calculating rolling volatility (default is 5).

        Returns:
        - bool: True if volatility clustering is observed, False otherwise.
        """
        data = self.data
        log_returns = np.diff(np.log(data))
        # Calculate rolling standard deviation (volatility)
        data2 = log_returns**2
        ljung_box_results = acorr_ljungbox(data2, lags=10, return_df=True)
        #plt.plot(data2)
        #plt.show()
        return all(ljung_box_results['lb_pvalue'] < 0.05 / 10) # Bonferroni correction
    
    def is_autocorrelated(self, lags: int = 10, significance_level: float = 0.05) -> bool:
        """
        Checks if a stock price time series exhibits autocorrelation.

        Parameters:
        - data (np.ndarray): 1D array of stock prices.
        - lags (int): Number of lags to test for autocorrelation (default is 10).
        - significance_level (float): The significance level for the test (default is 0.05).

        Returns:
        - bool: False if the series shows lack of autocorrelation, True otherwise.
        """
        data = self.data
        if len(data) < lags + 1:
            raise ValueError("Time series must have more data points than the number of lags.")
        
        log_returns = np.diff(np.log(data))
        
        # Test for autocorrelation using the Ljung-Box test
        ljung_box_results = acorr_ljungbox(log_returns, lags=lags, return_df=True)
        
        # Check if all p-values are above the significance level
        #print(ljung_box_results)
        return all(ljung_box_results['lb_pvalue'] < significance_level / lags) # Bonferroni correction

    def is_asymmetrical(self, significance_level: float = -0.5) -> bool:
        """
        Checks if a stock price time series exhibits asymmetry, where volatility increases more when prices fall compared to when they rise.

        Parameters:
        - data (np.ndarray): 1D array of stock prices or log returns.

        Returns:
        - bool: True if the series shows asymmetry, False otherwise.
        """
        data = self.data
        log_returns = np.diff(np.log(data))
        
        # Calculate the skewness of the log returns
        skewness = stats.skew(log_returns)

        return skewness < significance_level or skewness > -significance_level