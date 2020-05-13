from main import cointegrate # this is hacky as hell and requires modifying the cointegrate function's return values
import matplotlib.pyplot as plt
from os import path

with open("random.txt", "r") as random:
    p_values = []
    random_set = random.read().split(",")
        
    for pair in random_set:
        ticker1, ticker2 = pair.split("/")
        ticker1 = ticker1.strip()
        ticker2 = ticker2.strip()

        p_values.append(cointegrate(ticker1, ticker2))

plt.xlim(xmin = 0, xmax = 1)
plt.ylim(ymin = 0, ymax = 35)
plt.hist(p_values)
plt.title("Distribution of p-values in control set")
plt.ylabel("Frequency density")
plt.xlabel("p-value")
plt.savefig("../report/images/control_histogram.png", bbox_inches="tight")