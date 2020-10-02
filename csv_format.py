import pandas as pd
from glob import glob
from main import employee_type

employee_files = glob("experiments/employees/*/*.csv")
random_files = glob("experiments/random/*.csv")
quarters = ['2017Q2', '2017Q3', '2017Q4', '2018Q1']


def clean():
    for directory in employee_files:
        results = pd.read_csv(directory).set_index("pair")
        results.dropna(thresh=2, inplace=True)
        results.to_csv(directory)


def dissolved_fix():
    for directory in employee_files:
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


def count(files):
    for directory in files:
        survived = 0
        results = pd.read_csv(directory).set_index("pair")
        for i, row in results.iterrows():
            if str(row[0]) == "True" and str(row[1]) == "True":
                survived += 1

        print(directory)
        print("{}/{}".format(survived, results.iloc[:, 0].value_counts()[True]))
        print("#####")
