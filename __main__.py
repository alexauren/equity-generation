from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from data.fetch import FETCH
from discriminators.stylized_facts import STYLIZED_FACTS
from generators.brownian_motion import BROWNIAN_MOTION
from generators.sabr import SABR
from visualization.plotter import PLOTTER
import pprint

def simulate_brownian_motion(num_simulations, plot=False):
    """
    Simulates Brownian motion and plots the results.
    Parameters:
    num_simulations (int): The number of Brownian motion simulations to generate.
    """
    st, tt = BROWNIAN_MOTION().generate(M=num_simulations)
    if plot: 
        PLOTTER().plot(tt, st, title="Brownian Motion")
    return st

def simulate_sabr(plot=False):
    """
    Simulates a stock price using the SABR model and plots the results.
    """
    st, tt = SABR().generate()
    if plot:
        PLOTTER().plot(tt, st, title="SABR")
    return st

def discriminate_data(data):
    """
    Determines whether the data is real or synthetic based on the stylized facts.
    Parameters:
    data (np.ndarray): The data to discriminate.
    """
    sf = STYLIZED_FACTS(data)
    fat_tail = sf.check_fat_tail()
    volatility_clustering = sf.check_volatility_clustering()
    autocorrelation = sf.is_autocorrelated()
    asymmetric = sf.is_asymmetrical()
    return {'fat tail': fat_tail, 
            'volatility clustering': volatility_clustering, 
            'autocorrelation': autocorrelation, 
            'asymmetric': asymmetric}


# simulate_brownian_motion(1)
# data = simulate_sabr()
# data = simulate_sabr()
# FETCH().write(data, 'teamA_sabr', return_value=True)

def discriminate_return_data(filename):
    results = {}
    df = pd.read_csv(filename)
    for column in df:
            data = df[column]
            data = np.exp(np.cumsum(np.array(data)))
            results[column] = discriminate_data(data)
    return results

result = discriminate_return_data('data/csv/20241204mixeddata/mixed_teamC_sabr.csv')
pprint.pprint(result)
