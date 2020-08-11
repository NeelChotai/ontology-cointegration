import pandas as pd
from glob import glob

for directory in glob("experiments/*.csv"):
    raw_data = pd.read_csv(directory).set_index("pair")
    stay_cointegrated, gain_cointegration, lose_cointegration, stay_uncointegrated = 0, 0, 0, 0
    
    for index, row in raw_data.iterrows():
        cointegrated_2017 = row["cointegrated 2017"]
        cointegrated_2018 = row["cointegrated 2018"]

        if cointegrated_2017 == True and cointegrated_2018 == True:
            stay_cointegrated += 1
        elif cointegrated_2017 == False and cointegrated_2018 == True:
            gain_cointegration += 1
        elif cointegrated_2017 == True and cointegrated_2018 == False:
            lose_cointegration += 1
        elif cointegrated_2017 == False and cointegrated_2018 == False:
            stay_uncointegrated += 1

    directory = directory.replace("csv", "txt")
    with open(directory, "a") as output:
        output.write("Stayed cointegrated: {}\nGain cointegration: {}\nLose cointegration: {}\nStay uncointegrated: {}\n".format(
            stay_cointegrated, gain_cointegration, lose_cointegration, stay_uncointegrated))
