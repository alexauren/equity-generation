import pandas as pd
from alpha_vantage.timeseries import TimeSeries

class FETCH():

    def fetch(self, api_key:str, stock:str, size:str='compact') -> None:
        ts = TimeSeries(api_key, output_format='pandas')
        data, meta = ts.get_daily(stock, outputsize=size)
        self.write(data, stock)
    
    @staticmethod
    def write(data, filename: str) -> None:
        df = pd.DataFrame(data)
        df.to_csv(f"data/csv/{filename}")

FETCH().fetch("1MKFCLOIJI3QPW77", "SPY", 'full')
    
