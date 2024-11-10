import numpy as np
import scipy.stats as stats

class STYLIZED_FACTS:

    def __init__(self, data):
        self.data = data

    def check_fat_tail(data: np.ndarray, significance_level: float = 0.05) -> bool:
        """
        Checks if a stock price time series follows the fat tail phenomenon.

        Parameters:
        - data (np.ndarray): 1D array of stock prices.
        - significance_level (float): The significance level for the test (default is 0.05).

        Returns:
        - bool: True if the series follows the fat tail phenomenon, False otherwise.
        """

        log_returns = np.diff(np.log(data))
        
        # Fit the data to a normal distribution
        _, p_value = stats.kstest(log_returns, 'norm', args=(np.mean(log_returns), np.std(log_returns)))

        if p_value < significance_level:
            return True 
        else:
            return False 
        
    def check_volatility_clustering(data: np.ndarray, window_size: int = 5) -> bool:
        """
        Checks if a stock price time series exhibits volatility clustering.

        Parameters:
        - data (np.ndarray): 1D array of stock prices.
        - window_size (int): The number of periods for calculating rolling volatility (default is 5).

        Returns:
        - bool: True if volatility clustering is observed, False otherwise.
        """

        log_returns = np.diff(np.log(data))
        
        # Calculate rolling standard deviation (volatility)
        rolling_volatility = np.array([np.std(log_returns[i:i + window_size]) for i in range(len(log_returns) - window_size + 1)])

        # Check for clustering by comparing adjacent volatilities
        volatility_changes = np.diff(rolling_volatility)
        clustering = np.any(volatility_changes[:-1] * volatility_changes[1:] > 0)

        return clustering
    
    def check_lack_of_autocorrelation(data: np.ndarray, lags: int = 10, significance_level: float = 0.05) -> bool:
        """
        Checks if a stock price time series exhibits a lack of autocorrelation (i.e., if it is random).

        Parameters:
        - data (np.ndarray): 1D array of stock prices.
        - lags (int): Number of lags to test for autocorrelation (default is 10).
        - significance_level (float): The significance level for the test (default is 0.05).

        Returns:
        - bool: True if the series shows lack of autocorrelation, False otherwise.
        """

        if len(data) < lags + 1:
            raise ValueError("Time series must have more data points than the number of lags.")
        
        log_returns = np.diff(np.log(data))
        
        # Test for autocorrelation using the Ljung-Box test
        ljung_box_results = stats.acorr_ljungbox(log_returns, lags=lags, return_df=True)
        
        # Check if all p-values are above the significance level
        return all(ljung_box_results['lb_pvalue'] > significance_level)

    def check_asymmetry(data: np.ndarray) -> bool:
        """
        Checks if a stock price time series exhibits asymmetry, where volatility increases more when prices fall compared to when they rise.

        Parameters:
        - data (np.ndarray): 1D array of stock prices or log returns.

        Returns:
        - bool: True if the series shows asymmetry, False otherwise.
        """

        log_returns = np.diff(np.log(data))
        
        # Separate positive and negative returns
        positive_volatility = np.std(log_returns[log_returns > 0])
        negative_volatility = np.std(log_returns[log_returns < 0])

        return negative_volatility > positive_volatility