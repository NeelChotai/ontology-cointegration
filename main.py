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
from itertools import combinations

###
# the dates between which cointegration is tested
COINTEGRATION_START_DATE = "2016-07-01"
COINTEGRATION_END_DATE = "2018-04-01"
###

###
# the dates between which the tickers are tested for delisting and the number of trading days in this time period
BACKTESTING_START_DATE = COINTEGRATION_START_DATE
BACKTESTING_END_DATE = COINTEGRATION_END_DATE
TRADING_DAYS = 439
###


class coint_return(Enum):
    RELATIONSHIP = 0
    NO_RELATIONSHIP = 1
    INVALID = 2

class query_type(Enum):
    EMPLOYEE = 0
    DIRECTOR = 1
    DISTINCT = 2

class employee_type(Enum):
    EMPLOYEE = 0
    DIRECTOR = 1

GRAPH_CACHE = "/tmp/graph.cache"

def push_cache(cache_path, input_structure):
    outfile = open(cache_path, "wb")
    pickle.dump(input_structure, outfile)
    outfile.close()

def pop_cache(cache_path):
    infile = open(cache_path, "rb")
    cached = pickle.load(infile)
    infile.close()
    return cached

def populate():
    graph = rdflib.Graph()
    
    for x in range(83):
        graph.parse("data/ownership-{}.nt".format(str(x)), format="nt")
    
    return graph

def cointegrate(ticker1, ticker2):
    # cointegrates two time series given by tickers
    # return type: coint_return signifying relationship

    try:  # handle ticker not found errors
        if path.isfile("stocks/{}.csv".format(ticker1)):
            series1 = pd.read_csv("stocks/{}.csv".format(ticker1))
        else:
            series1 = yf.download(ticker1, period="10y").filter(
                ["Date", "Open"]).reset_index(drop=False)
            series1.to_csv("stocks/{}.csv".format(ticker1), index=False)

        if path.exists("stocks/{}.csv".format(ticker2)):
            series2 = pd.read_csv("stocks/{}.csv".format(ticker1))
        else:
            series2 = yf.download(ticker2, period="10y").filter(
                ["Date", "Open"]).reset_index(drop=False)
            series2.to_csv("stocks/{}.csv".format(ticker2), index=False)
    except:
        return coint_return.INVALID

    series1_backtest = (series1["Date"] >= BACKTESTING_START_DATE) & (
        series1["Date"] < BACKTESTING_END_DATE)
    series2_backtest = (series2["Date"] >= BACKTESTING_START_DATE) & (
        series2["Date"] < BACKTESTING_END_DATE)

    # ensures pairs we select can be backtested
    if len(series1.loc[series1_backtest]) < TRADING_DAYS or len(series2.loc[series2_backtest]) < TRADING_DAYS:
        return coint_return.INVALID

    merged = pd.merge(series1, series2, how="outer", on=["Date"])
    merged.dropna(inplace=True)
    merged.reset_index(inplace=True, drop=False)
    mask = (merged["Date"] >= COINTEGRATION_START_DATE) & (
        merged["Date"] < COINTEGRATION_END_DATE)
    merged = merged.loc[mask]

    if len(merged) < TRADING_DAYS:
        return coint_return.INVALID

    johansen_frame = pd.DataFrame(
        {"x": merged["Open_x"], "y": merged["Open_y"]})

    try:  # weird divide by zero error here on occassion
        score, p_value, _ = coint(merged["Open_x"], merged["Open_y"])
        johansen = coint_johansen(johansen_frame, 0, 1)
    except:
        return coint_return.INVALID

    # calculates cointegration using p-value and trace statistic (95% confidence)
    if p_value < 0.05 and johansen.cvt[0][1] < johansen.lr1[0]:
        return coint_return.RELATIONSHIP
    return coint_return.NO_RELATIONSHIP


def query(graph, type):
    # return type: list of tuples (SEC report URL, name, ticker1, ticker2)

    if type == query_type.DIRECTOR:
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
    elif type == query_type.EMPLOYEE:
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
    elif type == query_type.DISTINCT:
        query = graph.query(
            '''
            SELECT distinct ?t1
            WHERE { {
                ?company <http://york.ac.uk/tradingsymbol> ?t1 .
                } }
            ''') # returns all permutations of pairs in ontology

    return query


def clean(ticker):
    replace_with_blank = ["NASDAQ", "NYSE", "*", ":", "/", " ", "[", "]", '"']
    ticker = ticker.replace(".", "-")
    ticker = ticker.replace("CRDA CRDB", "CRDA")
    ticker = ticker.replace("CRDA-CRDB", "CRDB")
    ticker = ticker.replace('FCE-A/FCEB', "FCEA")
    ticker = ticker.replace('FCEA/FCEB', "FCEB")
    ticker = ticker.replace('BFA, BFB', "BFB")
    for symbol in replace_with_blank:
        ticker = ticker.replace(symbol, "")

    return ticker


def pair_people(query):
    # returns a dictionary of pairs and attributes

    pair_people_out = {}

    for row in query:
        sec_report, person, ticker1, ticker2 = row
        sec_report, person, ticker1, ticker2 = str(sec_report), str(
            person), clean(str(ticker1).upper()), clean(str(ticker2).upper())
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

    sorted_pairs = {k: v for k, v in sorted(
        pair_people_out.items(), key=lambda item: item[1], reverse=True)}  # sorts dictionary descending by number of attributes

    return sorted_pairs


def dump_pairs(query, type):
    # populates pairs directory with list of pairs/attribute

    sorted_pairs = pair_people(query)

    if type == employee_type.DIRECTOR:
        directory = "pairs/directors.txt"
    elif type == employee_type.EMPLOYEE:
        directory = "pairs/employees.txt"

    with open(directory, "w") as pairs_file:
        for key in sorted_pairs:
            pairs_file.write(
                "{}/{}: {}\n".format(key[0], key[1], sorted_pairs[key]))


def random_count():
    # creates a list of random pairs for testing

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


def pair_count(companies, type):
    # don't use this
    count = 0
    size = 0
    pair_set = []
    pair_coint = []
    pair_total = []
    pair_counts = []

    if type == employee_type.DIRECTOR: # output
        directory = "pairs/directors_cointegrated.txt"
    elif type == employee_type.EMPLOYEE:
        directory = "pairs/employees_cointegrated.txt" 

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

    for pair in pair_set:
        if cointegrate(pair[0], pair[1]) == coint_return.RELATIONSHIP:
            count += 1
            pair_coint.append(pair)
        pair_total.append(pair)

    for pair in pair_total:
        pair_counts.append(companies[pair])

    pair_counts = dict(pd.Series(pair_counts).value_counts())

    return count


def cointegrated_count(query, type):
    pair_coint = []
    sorted_pairs = pair_people(query)

    if type == employee_type.DIRECTOR:
        directory = "pairs/directors_cointegrated.txt"
    elif type == employee_type.EMPLOYEE:
        directory = "pairs/employees_cointegrated.txt"

    for pair in sorted_pairs:
        if pair[0] != pair[1] and cointegrate(pair[0], pair[1]) == coint_return.RELATIONSHIP:
            pair_coint.append(pair)

    for pair in pair_coint:
        with open(directory, "w") as pairs_file:
            for pair in pair_coint:
                pairs_file.write(
                    "{}/{}: {}\n".format(pair[0], pair[1], sorted_pairs[pair]))


def generate_set(size=30, companies=None):
    # generates set of pairs of random companies if no input dictionary specificied
    # otherwise, generates a set of pairs of companies from dictionary

    count = 0
    pair_set = []

    if companies == None:
        with open("stocks.txt") as stocks:
            tickers = stocks.read().split(",")
    else:
        pairs = list(companies)

    while True:
        pair = (choice(tickers), choice(tickers)
                ) if companies == None else choice(pairs)

        if pair[0] == pair[1] or pair in pair_set or cointegrate(pair[0], pair[1]) == coint_return.INVALID:
            count -= 1
        else:
            pair_set.append(pair)

        count += 1

        if count >= size:
            break

    return pair_set


def sampling(companies):
    # returns how many pairs are cointegrated given a list of company pairs

    total_cointegrated = 0

    for pair in companies:
        if cointegrate(pair[0], pair[1]) == coint_return.RELATIONSHIP:
            total_cointegrated += 1

    return total_cointegrated


def get_minimum_pairs(graph, num, samples, pairs):
    # sorted dictionaries of pair: attributes

    director_pairs = pair_people(query(graph, query_type.DIRECTOR))
    employee_pairs = pair_people(query(graph, query_type.EMPLOYEE))

    director_pairs = [x for x in list(
        director_pairs) if director_pairs[x] >= num]
    employee_pairs = [x for x in list(
        employee_pairs) if employee_pairs[x] >= num]

    director_sampling, employee_sampling = [], []

    for x in range(samples):  # 10 samples of 30 pairs
        director_set = generate_set(30, companies=director_pairs)
        employee_set = generate_set(30, companies=employee_pairs)

        director_sampling.append(sampling(director_set))
        employee_sampling.append(sampling(employee_set))

if path.isfile(GRAPH_CACHE):
    graph = pop_cache(GRAPH_CACHE)
else:
    graph = populate()
    push_cache(GRAPH_CACHE, graph)

companies = query(graph, query_type.DISTINCT)
companies_list = []

for row in companies:
    companies_list.append(row[0])

pair_combinations = combinations(companies_list, 2)
cointegrated_table = pd.DataFrame(columns=["pair", "cointegrated"]).set_index("pair")

for pair in pair_combinations:
    status = cointegrate(pair[0], pair[1])
    if status == coint_return.RELATIONSHIP:
        cointegrated_table = cointegrated_table.append({"pair": pair, "cointegrated": True})
    elif status == coint_return.NO_RELATIONSHIP:
        cointegrated_table = cointegrated_table.append({"pair": pair, "cointegrated": False})

cointegrated_table.to_csv("pairs/cointegrated_table.csv", index=False)

# next step is to go through cointegrated companies table and indicate the number of shared employees per pair