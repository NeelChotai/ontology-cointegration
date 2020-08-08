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
from glob import glob
from time import sleep


class coint_return(Enum):
    RELATIONSHIP = 0
    NO_RELATIONSHIP = 1
    INVALID = 2


class employee_type(Enum):
    EMPLOYEE = 0
    DIRECTOR = 1
    ALL = 2


###
# the dates between which cointegration is tested
COINTEGRATION_START_DATE = "2017-04-01"
COINTEGRATION_END_DATE = "2017-06-30"
TRADING_DAYS = 63
###

GRAPH_CACHE = "/tmp/graph.cache"
RANDOM_SET_SIZE = 50000


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

    for nt in glob("2017/*.nt"):
        graph.parse(nt, format="nt")

    return graph


def fetch_ticker(ticker):
    if path.isfile("stocks/{}.csv".format(ticker)):
        series = pd.read_csv("stocks/{}.csv".format(ticker))
        series["Date"] = series["Date"].apply(pd.to_datetime)
    else:
        try:
            series = yf.download(ticker, start=COINTEGRATION_START_DATE, end=COINTEGRATION_END_DATE,
                                 threads=False).filter(["Date", "Close"]).reset_index(drop=False)
            series.to_csv("stocks/{}.csv".format(ticker), index=False)
        except:
            return None
    return series


def cointegrate(ticker1, ticker2):
    # cointegrates two time series given by tickers
    # return type: coint_return signifying relationship

    series1 = fetch_ticker(ticker1)
    series2 = fetch_ticker(ticker2)

    try:
        merged = pd.merge(series1, series2, how="outer", on=["Date"])
    except:
        return (coint_return.INVALID, None)
    merged.dropna(inplace=True)
    merged.reset_index(inplace=True, drop=False)

    if len(merged) < TRADING_DAYS:
        return (coint_return.INVALID, None)

    johansen_frame = pd.DataFrame(
        {"x": merged["Close_x"], "y": merged["Close_y"]})

    try:  # weird divide by zero error here on occassion
        score, p_value, _ = coint(merged["Close_x"], merged["Close_y"])
        johansen = coint_johansen(johansen_frame, 0, 1)
    except:
        return (coint_return.INVALID, None)

    # calculates cointegration using p-value and trace statistic (95% confidence)
    if p_value < 0.05 and johansen.cvt[0][1] < johansen.lr1[0]:
        return (coint_return.RELATIONSHIP, p_value)
    return (coint_return.NO_RELATIONSHIP, p_value)


def query(graph, type):
    # return type: list of tuples (SEC report URL, name, ticker1, ticker2)

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
                ?quarter <http://york.ac.uk/periodreport> ?q .
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
    elif type == employee_type.ALL:
        query = graph.query(
            '''
            SELECT ?t
            WHERE { {
                ?company <http://york.ac.uk/tradingsymbol> ?t .
                } }
            ''')  # all companies in quarter

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


def pairs_with_attributes(query):
    # returns a dictionary of pairs and number attributes

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


def generate_attribute_set(companies):
    pair_set = []

    for pair in companies:
        reversed_pair = (pair[1], pair[0])
        result, p_value = cointegrate(pair[0], pair[1])

        if pair[0] == pair[1] or pair in pair_set or result == coint_return.INVALID or reversed_pair in pair_set:
            pass
        else:
            pair_set.append(pair)
    # pair_counts = dict(pd.Series(pair_counts).value_counts()) # number of attributes

    return pair_set


def generate_random_set(companies, size):
    # generates set of pairs of random companies of the length of the attribute set

    count = 0
    pair_set = []
    pairs = list(combinations(companies, 2))

    while True:
        pair = choice(pairs)
        reversed_pair = (pair[1], pair[0])
        result, p_value = cointegrate(pair[0], pair[1])

        if pair[0] == pair[1] or pair in pair_set or result == coint_return.INVALID or reversed_pair in pair_set:
            count -= 1
        else:
            pair_set.append(pair)

        count += 1

        if count >= size:
            break

    return pair_set


def cointegrated_count(pairs, type, interval):
    count = 0
    cointegrated = pd.DataFrame(
        columns=["pair", "cointegrated 2017", "p-value 2017"])

    for pair in pairs:
        result, p_value = cointegrate(pair[0], pair[1])
        if result == coint_return.RELATIONSHIP:
            cointegrated = cointegrated.append(
                {"pair": pair, "cointegrated 2017": True, "p-value 2017": p_value}, ignore_index=True)
            count += 1
        elif result == coint_return.NO_RELATIONSHIP:
            cointegrated = cointegrated.append(
                {"pair": pair, "cointegrated 2017": False, "p-value 2017": p_value}, ignore_index=True)

    cointegrated.set_index("pair", inplace=True)

    if type == employee_type.ALL:
        cointegrated.to_csv("experiment_1/random_q2.csv")
    elif type == employee_type.EMPLOYEE:
        cointegrated.to_csv(
            "experiment_1/employees_q2_interval_{}.csv".format(interval))

    return count


if path.isfile(GRAPH_CACHE):
    graph = pop_cache(GRAPH_CACHE)
else:
    graph = populate()
    push_cache(GRAPH_CACHE, graph)

companies_list = set()
for row in query(graph, employee_type.ALL):
    companies_list.add(str(row[0]))
random_pairs = generate_random_set(list(companies_list), RANDOM_SET_SIZE)

employee_dict = pairs_with_attributes(query(graph, employee_type.EMPLOYEE))

for interval in [2]:
    employee_pairs = [x for x in list(
        employee_dict) if employee_dict[x] >= interval]
    employee_pairs = generate_attribute_set(employee_pairs)

    with open("experiment_1/q2_2017_results.txt", "a") as results:
        results.write("Random set cointegrated: {}\n".format(
            cointegrated_count(random_pairs, employee_type.ALL, interval)))
        results.write("Employee set cointegrated ({} attribute(s)): {}\n".format(
            interval, cointegrated_count(employee_pairs, employee_type.EMPLOYEE, interval)))
        results.write("Total pairs in each set: {}\n".format(
            len(employee_pairs)))
