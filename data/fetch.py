import os
import pandas as pd
from alpha_vantage.timeseries import TimeSeries

class FETCH():

    @staticmethod
    def fetch(self, api_key:str, stock:str, size:str='compact') -> None:
        ts = TimeSeries(api_key, output_format='pandas')
        data, meta = ts.get_daily(stock, outputsize=size)
        self.write(data, stock)
    
    @staticmethod
    def write(data:tuple, filename: str, return_value=False) -> None:
        df = pd.DataFrame(data)
        if return_value:
            df = pd.DataFrame(df[0].pct_change())
            # df = df.drop(columns=[0])
        df.to_csv(f"data/csv/{filename}.csv")
        
    @staticmethod
    def split(data: str) -> None:
        df = pd.read_csv(f"data/csv/{data}.csv")
        close = df['4. close'].values
        num_series = len(close) // (252 * 5)
        
        for i in range(num_series):
            start_idx = i * 252 * 5
            end_idx = min(start_idx + (252 * 5), len(close))
            series = close[start_idx:end_idx]
            pd.Series(series).to_csv(f"data/csv/spy_series/{i+1}.csv")


# FETCH().fetch("1MKFCLOIJI3QPW77", "SPY", 'full')
# FETCH().split("SPY.csv")