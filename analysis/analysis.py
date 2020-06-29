import pandas as pd
import yfinance as yf
from statsmodels.tsa.stattools import coint
from scipy.stats import skew, kurtosis

def cointegrate(ticker1, ticker2):
    start_date = "2016-07-01"
    end_date = "2017-06-30"

    series1 = yf.download(ticker1, period="10y").filter(["Date", "Open"]).reset_index(drop=False)
    series2 = yf.download(ticker2, period="10y").filter(["Date", "Open"]).reset_index(drop=False)

    merged = pd.merge(series1, series2, how="outer", on=["Date"])
    merged.dropna(inplace=True)
    merged.reset_index(inplace=True, drop=False)
    mask = (merged["Date"] >= start_date) & (merged["Date"] < end_date)
    merged = merged.loc[mask]

    score, p_value, _ = coint(merged["Open_x"], merged["Open_y"])

    return p_value

p_values_control, p_values_directors, p_values_employees = [], [], []

with open("random.txt", "r") as random:
    random_set = random.read().split(",")
        
    for pair in random_set:
        ticker1, ticker2 = pair.split("/")
        ticker1 = ticker1.strip()
        ticker2 = ticker2.strip()

        p_values_control.append(cointegrate(ticker1, ticker2))

with open("directors.txt", "r") as directors:
    directors_set = directors.read().split(",")
        
    for pair in directors_set:
        ticker1, ticker2 = pair.split("/")
        ticker1 = ticker1.strip()
        ticker2 = ticker2.strip()

        p_values_directors.append(cointegrate(ticker1, ticker2))

with open("employees.txt", "r") as employees:
    employees_set = employees.read().split(",")
        
    for pair in employees_set:
        ticker1, ticker2 = pair.split("/")
        ticker1 = ticker1.strip()
        ticker2 = ticker2.strip()

        p_values_employees.append(cointegrate(ticker1, ticker2))

print("Skew of control: {}".format(skew(p_values_control)))
print("Kurtosis of control: {}".format(kurtosis(p_values_control)))
print("Skew of directors: {}".format(skew(p_values_directors)))
print("Kurtosis of directors: {}".format(kurtosis(p_values_directors)))
print("Skew of employees: {}".format(skew(p_values_employees)))
print("Kurtosis of employees: {}".format(kurtosis(p_values_employees)))