from os import path
from sys import stdout
from datetime import datetime
from random import choice
from enum import Enum
from statsmodels.tsa.stattools import coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import rdflib
import pickle

class coint_return(Enum):
    RELATIONSHIP = 0
    NO_RELATIONSHIP = 1
    INVALID = 2

cache_file = "/tmp/graph.cache"

def populate(cache_file):
    # for ease of programming
    # the graph data structure is enormous, pickling prevents regenerating the same structure every run

    if path.isfile(cache_file):
        infile = open(cache_file, "rb")
        graph = pickle.load(infile)
        infile.close()
    else:
        graph = rdflib.Graph()
        
        for x in range(29):
            graph.parse("data/ownership-{}.nt".format(str(x)), format="nt")
        
        outfile = open(cache_file, "wb")
        pickle.dump(graph, outfile)
        outfile.close()

    return graph

def query(graph):
    # SPARQL queries here
    # return format: dictionary where key = ticker pair and number of shared attributes
    # if sample size given, return subdict descending

    shared_attributes = graph.query(
        '''
        PREFIX foaf: <http://xmlns.com/foaf/0.1/>

        SELECT *
        WHERE {
            ?report foaf:name ?type .
        }
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

def random_count(size):
    count = 0
    random_set = []

    with open("stocks.txt") as stocks:
        tickers = stocks.read().split(",")
    
    for x in range(size): # preprocessing of control set
        pair = (choice(tickers), choice(tickers))
        reversed_pair = reversed(pair)
        result_random = cointegrate(pair[0], pair[1])

        if pair[0] == pair[1] or pair in random_set or result_random == coint_return.INVALID:
            sample_size += 1
        elif reversed_pair in random_set:
            result_existing = cointegrate(pair[1], pair[0]) # since this pair is in the set, coint_return will never be invalid
            
            if result_existing == coint_return.RELATIONSHIP:
                pass # if the existing entry has a cointegrating relationship, do nothing
            elif result_random == coint_return.RELATIONSHIP:
                random_set.remove(reversed_pair)
                random_set.append(pair)

            # only remaining possibility is neither pair has a cointegrating relationship
        else:
            random_set.append(pair)
            
    for pair in random_set: # testing cointegration of control set
        if cointegrate(pair[0], pair[1]) == coint_return.RELATIONSHIP:
            count += 1
        #stdout.write("{}/{}, ".format(pair[0], pair[1])) # to get results

    return count

def high_attribute_count(stocks):
    # preprocessing of high attribute set
    # input of stock pairs will be random, first: sort by descending key = relationship count
    # remove reflexive pairs
    # remove reversed pairs after testing
    # remove pairs with less than one year of history
    # if pair fails the above tests, increment preprocessing counter
    # otherwise, add pair a list of processed stocks
    # also check attributes and add to dictionary for distribution of relationships
    # once preprocessing counter hits 100, test for cointegration
    count = 0
    high_attribute_set = []

    for pair in high_attribute_set: # testing cointegration of high attribute set
        if cointegrate(pair[0], pair[1]) == coint_return.RELATIONSHIP:
            count += 1

    return count

def backtest(ticker1, ticker2):
    start_date = datetime(2015, 1, 1)
    end_date = datetime(2020, 1, 1)
    
    series1 = yf.download(ticker1, period="10y").filter(["Date", "Open"])
    series2 = yf.download(ticker2, period="10y").filter(["Date", "Open"])
    print(series1)
    merged = pd.merge(series1, series2, how="outer", on=["Date"])
    merged.dropna(inplace=True)
    merged.reset_index(inplace=True, drop=False)
    merged.rename(inplace=True, columns={"Open_x": ticker1, "Open_y": ticker2})
    mask = (merged["Date"] >= start_date) & (merged["Date"] < end_date)
    merged = merged.loc[mask]

    return merged

#graph = populate(cache_file)

#test = query(graph)
#for row in test:
#    print(row)
#    print("\n")

print(backtest("MA", "V"))