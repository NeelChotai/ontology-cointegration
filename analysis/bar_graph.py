import matplotlib.pyplot as plt
import numpy as np

directors = [1.3, 0.6, 1.1]
employees = [0.8, 1.6, 2.0]
random = 1.5

attr = [1, 3, 5]
pos = np.arange(len(attr))
width = 0.35

plt.title("Percentages of cointegrated pairs from sampling")
plt.bar(pos, directors, width, label = "Directors")
plt.bar(pos + width, employees, width, label = "Employees")
plt.xticks(pos + width / 2, attr)
plt.axhline(y = random, color="r")
plt.ylabel("Percentage of cointegrated pairs (average)")
plt.xlabel("Number of shared attributes (minimum)")
plt.legend(loc="best")
plt.savefig("../report/images/sampling.png", bbox_inches="tight")