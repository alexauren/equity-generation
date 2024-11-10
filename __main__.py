from generators.brownian_motion import BROWNIAN_MOTION
from generators.sabr import SABR
from visualization.plotter import PLOTTER

def simulate_brownian_motion(num_simulations):
    """
    Simulates Brownian motion and plots the results.
    Parameters:
    num_simulations (int): The number of Brownian motion simulations to generate.
    """
    st, tt = BROWNIAN_MOTION().generate(M=num_simulations)
    PLOTTER().plot(tt, st, title="Brownian Motion")

def simulate_sabr():
    """
    Simulates a stock price using the SABR model and plots the results.
    """
    st, tt = SABR().generate()
    PLOTTER().plot(tt, st, title="SABR")

simulate_brownian_motion(1)
simulate_sabr()
