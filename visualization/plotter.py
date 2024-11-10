import matplotlib.pyplot as plt
    
class PLOTTER: 

    @staticmethod
    def plot(tt, St, title="Stock Price"):    
        plt.plot(tt, St)
        plt.xlabel("Years $(t)$")
        plt.ylabel("Stock Price $(S_t)$")
        plt.title(f"Simulation of {title}")
        plt.show()