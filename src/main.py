from pandas import read_csv

def main():
    # after reading in ontology data, create a sorted list descending by highest cointegration factor
    # performance can be returned individually if user requests or generate graphs
    print("Hello world.")

def generate():
    # generate and sort SPARQL queries here
    # return as CSV?
    # return format: dictionary where key = ticker pair and cointegration score
    print("todo")

def evaluate(ticker):
    # read in ticker, output percentage increase within one month, 6 months and entire timeline
    # return percentage increase or some metric of performance
    data = read_csv("data/tickers/" + ticker + ".csv").sort_values("Date", ignore_index=True) # ticker value sorted by date
    close_price = data.filter(["Close"])
    one_month = ((close_price.loc[30] - close_price.loc[0])/close_price.loc[0]) * 100
    six_months = ((close_price.loc[182] - close_price.loc[0])/close_price.loc[0]) * 100
    timeline = ((close_price.loc[len(data) - 1] - close_price.loc[0])/close_price.loc[0]) * 100

    return float(one_month), float(six_months), float(timeline)
    

print(evaluate("INTC"))