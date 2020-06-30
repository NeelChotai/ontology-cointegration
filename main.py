from os import path
from sys import stdout
from random import choice
from enum import Enum
from statsmodels.tsa.stattools import coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import statsmodels.api as sm
import rdflib
import pickle


class coint_return(Enum):
    RELATIONSHIP = 0
    NO_RELATIONSHIP = 1
    INVALID = 2

class employee_type(Enum):
    EMPLOYEE = 0
    DIRECTOR = 1

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

        for x in range(83):
            graph.parse("data/ownership-{}.nt".format(str(x)), format="nt")

        outfile = open(cache_file, "wb")
        pickle.dump(graph, outfile)
        outfile.close()

    return graph


def cointegrate(ticker1, ticker2):
    # cointegrates two time series given by tickers

    start_date = "2016-07-01"
    end_date = "2017-06-30"
    start_date_backtest = "2017-07-01"
    end_date_backtest = "2018-06-30"

    try:  # handle ticker not found errors
        series1 = yf.download(ticker1, period="10y").filter(
            ["Date", "Open"]).reset_index(drop=False)
        series2 = yf.download(ticker2, period="10y").filter(
            ["Date", "Open"]).reset_index(drop=False)
    except:
        return coint_return.INVALID

    series1_backtest = (series1["Date"] >= start_date_backtest) & (
        series1["Date"] < end_date_backtest)
    series2_backtest = (series2["Date"] >= start_date_backtest) & (
        series2["Date"] < end_date_backtest)

    # ensures pairs we select can be backtested
    if len(series1.loc[series1_backtest]) < 251 or len(series2.loc[series2_backtest]) < 251:
        return coint_return.INVALID

    merged = pd.merge(series1, series2, how="outer", on=["Date"])
    merged.dropna(inplace=True)
    merged.reset_index(inplace=True, drop=False)
    mask = (merged["Date"] >= start_date) & (merged["Date"] < end_date)
    merged = merged.loc[mask]

    if len(merged) < 251:
        return coint_return.INVALID

    johansen_frame = pd.DataFrame(
        {"x": merged["Open_x"], "y": merged["Open_y"]})

    try:  # weird divide by zero error here on occassion
        score, p_value, _ = coint(merged["Open_x"], merged["Open_y"])
        johansen = coint_johansen(johansen_frame, 0, 1)
    except:
        return coint_return.INVALID

    # return p_value
    # calculates cointegration using p-value and trace statistic (95% confidence)
    if p_value < 0.05 and johansen.cvt[0][1] < johansen.lr1[0]:
        return coint_return.RELATIONSHIP
    return coint_return.NO_RELATIONSHIP


def query(graph, type):
    if type == employee_type.DIRECTOR:
        query = graph.query(
            '''
            SELECT ?person ?p ?t1 ?t2
            WHERE { {
                ?person <http://york.ac.uk/worksat> ?company .
                ?person <http://york.ac.uk/isdirector>  true .
                ?person <http://york.ac.uk/worksat> ?othercompany .
                ?person <http://xmlns.com/foaf/0.1/name> ?p .
                ?company <http://york.ac.uk/tradingsymbol> ?t1 .
                ?othercompany <http://york.ac.uk/tradingsymbol> ?t2 .
                FILTER(?t1 != ?t2)
                } }
            ''')  # returns pairs of companies and person
    elif type == employee_type.EMPLOYEE:
        query = graph.query(
            '''
            SELECT ?person ?p ?t1 ?t2
            WHERE { {
                ?person <http://york.ac.uk/worksat> ?company .
                ?person <http://york.ac.uk/worksat> ?othercompany .
                ?person <http://xmlns.com/foaf/0.1/name> ?p .
                ?company <http://york.ac.uk/tradingsymbol> ?t1 .
                ?othercompany <http://york.ac.uk/tradingsymbol> ?t2 .
                FILTER(?t1 != ?t2)
                } }
            ''')  # returns pairs of companies and person

    return query

def pair_people(query):
    pair_people_out = {}

    for row in query:
        sec_report, person, ticker1, ticker2 = row
        sec_report, person, ticker1, ticker2 = str(sec_report), str(
            person), str(ticker1).upper(), str(ticker2).upper()

        replace_with_blank = ["[", "]", '"', "NASDAQ", "NYSE: ", "NYSE/", "*", ":"]

        for ticker in [ticker1, ticker2]:
            ticker = ticker.replace(".", "-")  # for yf.download formatting
            ticker = ticker.replace("CRDA CRDB", "CRDA")
            ticker = ticker.replace("CRDA-CRDB", "CRDB")
            ticker = ticker.replace('FCE-A/FCEB', "FCEA")
            ticker = ticker.replace('FCEA/FCEB', "FCEB")
            ticker = ticker.replace('BFA, BFB', "BFB")
            for symbol in replace_with_blank:
                ticker = ticker.replace(symbol, "")

        pair = (ticker1, ticker2)

        if pair in pair_people_out:
            pair_people_out[pair].append(person)
        else:
            pair_people_out[pair] = [person]

    remove_keys = []
    manual = ["NONE", "N/A", "CRDA  CRDB"]
    for key in pair_people_out:
        key_reversed = (key[1], key[0])

        # reflexive pairs and manual filtering
        if key[0] == key[1] or key[0] in manual or key[1] in manual:
            remove_keys.append(key)
        elif key_reversed in pair_people_out:  # transitive pairs
            if key in remove_keys:
                pass
            else:
                remove_keys.append(key_reversed)

    for key in remove_keys:
        pair_people_out.pop(key)

    for key in pair_people_out:
        people = list(set(pair_people_out[key]))  # remove duplicate people
        pair_people_out[key] = len(people)  # amount of attributes

    # sorts dictionary descending by number of attributes
    sorted_pairs = {k: v for k, v in sorted(
        pair_people_out.items(), key=lambda item: item[1], reverse=True)}
    return sorted_pairs

def random_count():
    count = 0
    size = 0
    random_set = []
    random_coint = []
    random_total = []

    with open("stocks.txt") as stocks:
        tickers = stocks.read().split(",")

    while True:  # preprocessing of control set
        pair = (choice(tickers), choice(tickers))
        reversed_pair = reversed(pair)
        result_random = cointegrate(pair[0], pair[1])

        if pair[0] == pair[1] or pair in random_set or result_random == coint_return.INVALID:
            size -= 1
        elif reversed_pair in random_set:
            # since this pair is in the set, coint_return will never be invalid
            result_existing = cointegrate(pair[1], pair[0])

            if result_existing == coint_return.RELATIONSHIP:
                pass  # if the existing entry has a cointegrating relationship, do nothing
            elif result_random == coint_return.RELATIONSHIP:
                random_set.remove(reversed_pair)
                random_set.append(pair)
            # only remaining possibility is neither pair has a cointegrating relationship
        else:
            random_set.append(pair)

        size += 1

        if size >= 150:
            break

    for pair in random_set:  # testing cointegration of control set
        if cointegrate(pair[0], pair[1]) == coint_return.RELATIONSHIP:
            count += 1
            random_coint.append(pair)
        random_total.append(pair)

    with open("pairs/random.txt", "a") as random_output:
        random_output.write("Pairs:\n")
        for pair in random_total:
            random_output.write("{}/{}, ".format(pair[0], pair[1]))
        random_output.write("\nPairs with a cointegrating relationship:\n")
        for pair in random_coint:
            random_output.write("{}/{}, ".format(pair[0], pair[1]))

    return count


def pair_count(companies):
    count = 0
    size = 0
    pair_set = []
    pair_coint = []
    pair_total = []
    pair_counts = []

    for pair in companies:
        reversed_pair = (pair[1], pair[0])
        result_pair = cointegrate(pair[0], pair[1])

        # yes, this level of micromanagement is necessary
        if pair[0] == pair[1] or pair in pair_set or result_pair == coint_return.INVALID:
            size -= 1
        elif reversed_pair in pair_set:
            # since this pair is in the set, coint_return will never be invalid
            result_existing = cointegrate(pair[1], pair[0])

            if result_existing == coint_return.RELATIONSHIP:
                pass  # if the existing entry has a cointegrating relationship, do nothing
            elif result_pair == coint_return.RELATIONSHIP:
                pair_set.remove(reversed_pair)
                pair_set.append(pair)
            # only remaining possibility is neither pair has a cointegrating relationship
        else:
            pair_set.append(pair)

        size += 1

        if size >= 150:
            break

    for pair in pair_set:  # testing cointegration of control set
        if cointegrate(pair[0], pair[1]) == coint_return.RELATIONSHIP:
            count += 1
            pair_coint.append(pair)
        pair_total.append(pair)

    for pair in pair_total:
        pair_counts.append(companies[pair])

    pair_counts = dict(pd.Series(pair_counts).value_counts())

    with open("pairs/employees.txt", "a") as pair_output:
        pair_output.write("Pairs:\n")
        for pair in pair_total:
            pair_output.write("{}/{}, ".format(pair[0], pair[1]))
        pair_output.write("\nPairs with a cointegrating relationship:\n")
        for pair in pair_coint:
            pair_output.write("{}/{}, ".format(pair[0], pair[1]))
        pair_output.write(
            "\nDistribution of attributes (number of attributes: frequency):\n")
        for key in pair_counts:
            pair_output.write("{}/{}, ".format(key, pair_counts[key]))
    return count


def generate_random_set(size):
    count = 0
    random_set = []

    with open("stocks.txt") as stocks:
        tickers = stocks.read().split(",")

    while True:
        pair = (choice(tickers), choice(tickers))

        if pair[0] == pair[1] or pair in random_set or cointegrate(pair[0], pair[1]) == coint_return.INVALID:
            count -= 1
        else:
            random_set.append(pair)

        count += 1

        if count >= size:
            break

    return random_set


def generate_test_set(input_dict, size):
    count = 0
    pairs = list(input_dict)
    pair_set = []

    while True:
        pair = choice(pairs)

        if pair[0] == pair[1] or pair in pair_set or cointegrate(pair[0], pair[1]) == coint_return.INVALID:
            size -= 1
        else:
            pair_set.append(pair)

        count += 1

        if count >= size:
            break

    return pair_set


def sampling(companies):
    total_cointegrated = 0

    for pair in companies:
        if cointegrate(pair[0], pair[1]) == coint_return.RELATIONSHIP:
            total_cointegrated += 1

    return total_cointegrated

def get_minimum_pairs(graph, num, samples, pairs):
    director_pairs = pair_people(query(graph, employee_type.DIRECTOR)) # sorted dictionaries of pair: attributes
    employee_pairs = pair_people(query(graph, employee_type.EMPLOYEE))

    director_pairs = [x for x in list(director_pairs) if director_pairs[x] >= num]
    employee_pairs = [x for x in list(employee_pairs) if employee_pairs[x] >= num]

    director_sampling, employee_sampling = [], []
    
    for x in range(samples):  # 10 samples of 30 pairs
        director_set = generate_test_set(director_pairs, pairs)
        employee_set = generate_test_set(employee_pairs, pairs)
        
        director_sampling.append(sampling(director_set))
        employee_sampling.append(sampling(employee_set))

graph = populate(cache_file)

print("Director set average: {}".format(np.mean(director_sampling)))
print("Director set: {}".format(director_sampling))
print("Employee set average: {}".format(np.mean(employee_sampling)))
print("Employee set: {}".format(employee_sampling))
