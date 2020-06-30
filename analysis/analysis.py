import pandas as pd
import yfinance as yf
from statsmodels.tsa.stattools import coint
from scipy.stats import skew, kurtosis
import matplotlib.pyplot as plt

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

def p_value_aggregate(input_set):
    p_values_out = []

    for pair in input_set:
        ticker1, ticker2 = pair.split("/")
        ticker1 = ticker1.strip()
        ticker2 = ticker2.strip()

        p_values_out.append(cointegrate(ticker1, ticker2))
    
    return p_values_out

with open("random.txt", "r") as random:
    random_set = random.read().split(",")
    p_values_control = p_value_aggregate(random_set)

with open("directors.txt", "r") as directors:
    directors_set = directors.read().split(",")
    p_values_directors = p_value_aggregate(directors_set)

with open("employees.txt", "r") as employees:
    employees_set = employees.read().split(",")
    p_values_employees = p_value_aggregate(employees_set)

for p_values, name in [(p_values_control, "control"), (p_values_directors, "directors"), (p_values_employees, "employees")]:
    plt.xlim(xmin = 0, xmax = 1)
    plt.ylim(ymin = 0, ymax = 35)
    plt.hist(p_values)
    plt.title("Distribution of p-values in {} set".format(name))
    plt.ylabel("Frequency")
    plt.xlabel("p-value")
    plt.savefig("{}_histogram.png".format(name), bbox_inches="tight")

print("Skew of control: {}".format(skew(p_values_control)))
print("Kurtosis of control: {}".format(kurtosis(p_values_control)))
print("Skew of directors: {}".format(skew(p_values_directors)))
print("Kurtosis of directors: {}".format(kurtosis(p_values_directors)))
print("Skew of employees: {}".format(skew(p_values_employees)))
print("Kurtosis of employees: {}".format(kurtosis(p_values_employees)))