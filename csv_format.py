import pandas as pd
from glob import glob

files = glob("experiments/*.csv")
quarters = ['2017Q2', '2017Q3', '2017Q4', '2018Q1']

def clean():
    for directory in files:
        results = pd.read_csv(directory).set_index("pair")
        results.dropna(thresh=2, inplace=True)
        results.to_csv(directory)

def dissolved_fix():
    for directory in files:
        results = pd.read_csv(directory).set_index("pair")
        for i, row in results.iterrows():
            if row['2017Q3'] == "DISSOLVED":
                results.at[i, '2017Q4'] = "DISSOLVED"
                results.at[i, '2018Q1'] = "DISSOLVED"
                results.at[i, '2018Q2'] = "DISSOLVED"
            elif row['2017Q4'] == "DISSOLVED":
                results.at[i, '2018Q1'] = "DISSOLVED"
                results.at[i, '2018Q2'] = "DISSOLVED"
            elif row['2018Q1'] == "DISSOLVED":
                results.at[i, '2018Q2'] = "DISSOLVED"
        results.to_csv(directory)

def count():
    for directory in files:
        survived, total = [0, 0, 0, 0], [0, 0, 0, 0]
        results = pd.read_csv(directory).set_index("pair")
        for i, row in results.iterrows():
            if str(row['2017Q2']) == "True" and str(row['2017Q3']) == "True":
                survived[0] += 1
            if str(row['2017Q3']) == "True" and str(row['2017Q4']) == "True":
                survived[1] += 1
            if str(row['2017Q4']) == "True" and str(row['2018Q1']) == "True":
                survived[2] += 1
            if str(row['2018Q1']) == "True" and str(row['2018Q2']) == "True":
                survived[3] += 1

        count = 0
        for q in quarters:
            total[count] = len(results[q].dropna())
            count += 1

        print(directory)
        for s, t in zip(survived, total):
            print("{}/{}".format(s, t))
        print("#####")

count()