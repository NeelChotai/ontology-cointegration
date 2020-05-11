import os
import sys
from datetime import datetime
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen
import rdflib
import pickle
import yfinance as yf
from enum import Enum

class coint_return(Enum):
    RELATIONSHIP = 0
    NO_RELATIONSHIP = 1
    INVALID = 2

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

    return graph

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

    start_date = datetime(2015, 1, 1)
    end_date = datetime(2020, 1, 1)

    try: # handle ticker not found errors
        series1 = yf.download(ticker1, period="10y").filter(["Date", "Open"])
        series2 = yf.download(ticker2, period="10y").filter(["Date", "Open"])
    except:
        return coint_return.INVALID
    
    merged = pd.merge(series1, series2, how="outer", on=["Date"])
    merged.dropna(inplace=True)
    merged.reset_index(inplace=True, drop=False)
    mask = (merged["Date"] >= start_date) & (merged["Date"] < end_date)
    merged = merged.loc[mask]

    if len(merged) <= 365:
        return coint_return.INVALID

    johansen_frame = pd.DataFrame({"x":merged["Open_x"], "y":merged["Open_y"]})
    score, p_value, _ = coint(merged["Open_x"], merged["Open_y"])
    johansen = coint_johansen(johansen_frame, 0, 1)

    if p_value < 0.05 and johansen.cvt[0][0] < johansen.lr1[0]: # calculates cointegration using p-value and trace statistic (90% confidence)
        print(ticker1, ticker2) # debug
        return coint_return.RELATIONSHIP
    return coint_return.NO_RELATIONSHIP

def random_companies(sample_size):
    count = 0
    random_set = []

    with open("stocks.txt") as stocks:
        tickers = stocks.read().split(",")
    
    for x in range(sample_size):
        pair = (random.choice(tickers), random.choice(tickers))
        
        if pair[0] == pair[1] or pair in random_set:
            sample_size += 1
        elif reversed(pair) in random_set:
            result = cointegrate(pair[0], pair[1])
            result_reversed = cointegrate(pair[1], pair[0])
        else:
            result = cointegrate(pair[0], pair[1])
            if result == coint_return.INVALID:
                sample_size += 1
            else:
                if result == coint_return.RELATIONSHIP:
                    count += 1
                random_set.append(pair)
                #sys.stdout.write("{}/{}, ".format(pair[0], pair[1]))

    return count

def preprocess(stocks):
    # remove reflexive pairs
    # remove reversed pairs after testing
    # remove pairs with less than one year of history
    
    return stocks

def cointegration_count(stocks): # this will probably just be appended to the end
    for pair in stocks:
        if cointegrate(pair[0], pair[1]) == coint_return.RELATIONSHIP:
            count += 1
            #sys.stdout.write("{}/{}, ".format(pair[0], pair[1]))

    return count

#graph = initial_run(cache_file)

print(random_companies(100))