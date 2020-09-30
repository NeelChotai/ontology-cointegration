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
from ast import literal_eval


class coint_return(Enum):
    RELATIONSHIP = 0
    NO_RELATIONSHIP = 1
    INVALID = 2


class employee_type(Enum):
    EMPLOYEE = 0
    DIRECTOR = 1
    ALL = 2


class Quarter:
    def __init__(self, start, end, days, folder):
        self.start = start
        self.end = end
        self.days = days
        self.folder = folder


# constants
Q22017 = Quarter("2017-04-01", "2017-06-30", 63, "2017Q2")
Q32017 = Quarter("2017-07-01", "2017-09-30", 64, "2017Q3")
Q42017 = Quarter("2017-10-01", "2017-12-31", 63, "2017Q4")
Q12018 = Quarter("2018-01-01", "2018-03-31", 61, "2018Q1")
Q22018 = Quarter("2018-04-01", "2018-06-30", 64, "2018Q2")
OBJECT_LIST = [Q32017, Q42017, Q12018, Q22018]
GRAPH_CACHE = "/tmp/graph.cache"
RANDOM_SET_SIZE = 50000
###


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

    for nt in glob("data/{}/*.nt".format(QUARTER)):
        graph.parse(nt, format="nt")

    return graph


def fetch_ticker(ticker):
    directory = "stocks/{}/{}.csv".format(QUARTER, ticker)
    if path.isfile(directory):
        series = pd.read_csv(directory)
        series["Date"] = series["Date"].apply(pd.to_datetime)
    else:
        try:
            series = yf.download(ticker, start=COINTEGRATION_START_DATE, end=COINTEGRATION_END_DATE,
                                 threads=False).filter(["Date", "Close"]).reset_index(drop=False)
            series.to_csv(directory, index=False)
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


def cointegrated_count_first(pairs, type, interval):
    count = 0
    generated_pairs = None

    if type == employee_type.ALL:
        directory = "experiments/random.csv"
    elif type == employee_type.EMPLOYEE:
        directory = "experiments/employees_interval_{}.csv".format(interval)

    if path.isfile(directory):
        cointegrated = pd.read_csv(directory).set_index("pair")
        new_cointegrated = pd.DataFrame(columns=["pair", QUARTER])
        if pairs == None:
            pairs = [literal_eval(p) for p in cointegrated.index.values]
        else:
            existing_pairs = [literal_eval(p)
                              for p in cointegrated.index.values]
            generated_pairs = set(pairs)
            generated_pairs.update(existing_pairs)
            generated_pairs = list(generated_pairs)
    else:
        cointegrated = pd.DataFrame(columns=["pair", QUARTER]) # this will not work in its current form

    for pair in generated_pairs:
        result, p_value = cointegrate(pair[0], pair[1])
        formatted_pair = str(pair)
        if result == coint_return.RELATIONSHIP:
            if formatted_pair in cointegrated.index.values:
                cointegrated.loc[formatted_pair, QUARTER] = True
            else:
                new_cointegrated = new_cointegrated.append(
                    {"pair": formatted_pair, QUARTER: True}, ignore_index=True)

            if pair in pairs:
                count += 1
        elif result == coint_return.NO_RELATIONSHIP:
            if formatted_pair in cointegrated.index.values:
                cointegrated.loc[formatted_pair, QUARTER] = False
            else:
                new_cointegrated = new_cointegrated.append(
                    {"pair": formatted_pair, QUARTER: False}, ignore_index=True)
        else:
            if formatted_pair in cointegrated.index.values:
                cointegrated.loc[formatted_pair, QUARTER] = "DISSOLVED"
            else:
                new_cointegrated = new_cointegrated.append(
                    {"pair": formatted_pair, QUARTER: "DISSOLVED"}, ignore_index=True)

    new_cointegrated.set_index("pair", inplace=True)
    cointegrated = cointegrated.append(new_cointegrated)
    cointegrated.to_csv(directory)

    return count


def cointegrated_count_last(directory):  # todo: fix this
    count = 0
    cointegrated = pd.read_csv(directory).set_index("pair")
    pairs = cointegrated.index.values

    for pair in pairs:
        tuple_pair = literal_eval(pair)
        result, p_value = cointegrate(tuple_pair[0], tuple_pair[1])
        if result == coint_return.RELATIONSHIP:
            cointegrated.loc[pair, "cointegrated 2019"] = True
            cointegrated.loc[pair, "p-value 2019"] = p_value
            count += 1
        else:
            cointegrated.loc[pair, "cointegrated 2019"] = False
            cointegrated.loc[pair, "p-value 2019"] = p_value

    cointegrated.to_csv(directory)

    return (count, len(cointegrated))


def sliding_first(type, pairs, interval=None):
    if type == employee_type.ALL:
        directory = "experiments/random/random_{}.csv".format(QUARTER)
    elif type == employee_type.EMPLOYEE:
        directory = "experiments/employees/interval_{}/employee_{}_{}.csv".format(
            interval, interval, QUARTER)

    cointegrated = pd.DataFrame(columns=["pair", QUARTER]).set_index("pair")

    for pair in list(pairs):
        result, p_value = cointegrate(pair[0], pair[1])
        formatted_pair = str(pair)
        if result == coint_return.RELATIONSHIP:
            cointegrated.loc[formatted_pair, QUARTER] = True
        elif result == coint_return.NO_RELATIONSHIP:
            cointegrated.loc[formatted_pair, QUARTER] = False
        else:
            cointegrated.loc[formatted_pair, QUARTER] = "DISSOLVED"

    cointegrated.to_csv(directory)


def sliding_last(type, previous_quarter, interval=None):
    if type == employee_type.ALL:
        directory = "experiments/random/random_{}.csv".format(previous_quarter)
    elif type == employee_type.EMPLOYEE:
        directory = "experiments/employees/interval_{}/employee_{}_{}.csv".format(
            interval, interval, previous_quarter)

    cointegrated = pd.read_csv(directory).set_index("pair")
    pairs = [literal_eval(p) for p in cointegrated.index.values]

    for pair in pairs:
        result, p_value = cointegrate(pair[0], pair[1])
        formatted_pair = str(pair)
        if result == coint_return.RELATIONSHIP:
            cointegrated.loc[formatted_pair, QUARTER] = True
        elif result == coint_return.NO_RELATIONSHIP:
            cointegrated.loc[formatted_pair, QUARTER] = False
        else:
            cointegrated.loc[formatted_pair, QUARTER] = "DISSOLVED"

    cointegrated.to_csv(directory)


def get_companies(graph):
    companies_list = set()
    for row in query(graph, employee_type.ALL):
        companies_list.add(str(row[0]))
    random_pairs = generate_random_set(list(companies_list), RANDOM_SET_SIZE)

    return random_pairs


def generate_employee_results():
    for obj in OBJECT_LIST:
        global COINTEGRATION_START_DATE
        global COINTEGRATION_END_DATE
        global TRADING_DAYS
        global QUARTER

        COINTEGRATION_START_DATE = obj.start
        COINTEGRATION_END_DATE = obj.end
        TRADING_DAYS = obj.days
        QUARTER = obj.folder

        graph = populate()
        employee_dict = pairs_with_attributes(
            query(graph, employee_type.EMPLOYEE))

        cointegrated_count_first(None, employee_type.ALL, None)

        with open("experiments/output/{}.txt".format(QUARTER), "a") as results:
            for interval in range(1, 6):
                employee_pairs = [x for x in list(
                    employee_dict) if employee_dict[x] >= interval]
                employee_pairs = generate_attribute_set(employee_pairs)

                results.write("\nEmployee set cointegrated ({} attribute(s)): {}\n".format(
                    interval, cointegrated_count_first(employee_pairs, employee_type.ALL, interval)))
                results.write("Total pairs in employee set: {}\n".format(
                    len(employee_pairs)))


def generate_survival(type):
    previous_quarter = None
    for obj in OBJECT_LIST:
        global COINTEGRATION_START_DATE
        global COINTEGRATION_END_DATE
        global TRADING_DAYS
        global QUARTER

        COINTEGRATION_START_DATE = obj.start
        COINTEGRATION_END_DATE = obj.end
        TRADING_DAYS = obj.days
        QUARTER = obj.folder

        graph = populate()
        if type == employee_type.ALL:
            companies_list = set()
            for row in query(graph, employee_type.ALL):
                companies_list.add(clean(str(row[0]).upper()))
            random_pairs = generate_random_set(
                list(companies_list), RANDOM_SET_SIZE)

            sliding_first(type, random_pairs)
            if previous_quarter is not None:
                sliding_last(type, previous_quarter)
            previous_quarter = QUARTER

        elif type == employee_type.EMPLOYEE:
            employee_dict = pairs_with_attributes(
                query(graph, employee_type.EMPLOYEE))

            for interval in range(1, 6):
                employee_pairs = [x for x in list(
                    employee_dict) if employee_dict[x] >= interval]
                employee_pairs = generate_attribute_set(employee_pairs)

                sliding_first(type, employee_pairs, interval=interval)
                if previous_quarter is not None:
                    sliding_last(type, previous_quarter, interval=interval)
            previous_quarter = QUARTER


generate_survival(employee_type.EMPLOYEE)