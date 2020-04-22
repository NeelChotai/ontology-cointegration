from pandas import read_csv
import rdflib
import os

def preprocess(graph):
    #remove entries for which we don't have ticker data
    tickers = []

    for filename in os.listdir("data/tickers/"):
        ticker = filename.split(".")
        tickers.append(ticker[0])

    # scan through ontology, check if each ticker value exists in data
    # if not, remove entry
    # return transformed graph

    return graph

def populate():
    graph = rdflib.Graph()

    for x in range(0, 29):
        temp_graph = rdflib.Graph().parse("data/directorship/ownership-" + str(x) + ".nt", format="nt")
        graph += temp_graph

    return preprocess(graph)

def generate():
    # generate and sort SPARQL queries here
    # return as CSV?
    # return format: dictionary where key = ticker pair and number of shared attributes
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

graph = populate()