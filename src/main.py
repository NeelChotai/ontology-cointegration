import pandas as pd
import numpy as np
import rdflib
import os
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen
import random
import pickle
import yfinance as yf

cache_file = "/tmp/graph.cache"

def populate():
    graph = rdflib.Graph()

    for x in range(0, 29):
        temp_graph = rdflib.Graph().parse("data/directorship/ownership-{}.nt".format(str(x)), format="nt")
        graph += temp_graph

    return graph

def initial_run(cache_file):
    # for ease of programming
    # the graph data structure is enormous, pickling prevents regenerating the same structure every run

    if os.path.isfile(cache_file):
        infile = open(cache_file, "rb")
        graph = pickle.load(infile)
        infile.close()
    else:
        graph = populate()
        outfile = open(cache_file, "wb")
        pickle.dump(graph, outfile)
        outfile.close()

def random_companies(sample_size):
    random_out = []

    with open("stocks.txt") as stocks:
        tickers = stocks.read().split(",")
    
    for x in range(sample_size):
        pair = (random.choice(tickers), random.choice(tickers))
        
        if pair[0] == pair[1] or pair in random_out or reversed(pair) in random_out:
            sample_size += 1
        else:
            random_out.append(pair)

    return random_out

def generate(graph):
    # SPARQL queries here
    # return format: dictionary where key = ticker pair and number of shared attributes
    # if sample size given, return subdict descending

    shared_attributes = graph.query(
        ''' some query
        ''')
    
    return shared_attributes

def cointegrate(ticker1, ticker2):
    # cointegrates two time series given by tickers
    # returns True if time series are cointegrated

    series1 = yf.download(ticker1, "2015-01-01", "2020-01-01").filter(["Date", "Close"])
    series2 = yf.download(ticker2, "2015-01-01", "2020-01-01").filter(["Date", "Close"])
    
    merged = pd.merge(series1, series2, how="outer", on=["Date"])
    merged.dropna(inplace=True)
    print(merged)
    johansen_frame = pd.DataFrame({"x":merged["Close_x"], "y":merged["Close_y"]})

    score, p_value, _ = coint(merged["Close_x"], merged["Close_y"])
    johansen = coint_johansen(johansen_frame, 0, 1)

    if p_value < 0.05 or johansen.cvt[1][0] < johansen.lr1[1]: # calculates cointegration using p-value and trace statistic (90% confidence)
        return True
    return False

#test = random_companies(100)

#import sys
#for pair in test:
#    sys.stdout.write("{}/{}, ".format(pair[0], pair[1]))

p = cointegrate("GOOG", "GOOGL")