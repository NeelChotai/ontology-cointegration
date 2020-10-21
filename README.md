# Financial ontology algorithms

Project files for CIFEr paper.

One big inefficiency is generating the cointegration result twice: once to validate the pair, another to log to the CSV. The performance impact is rather negligible since the bottleneck is fetching data (which is cached) and the cointegration tests are just simple equations, however this should be easy to amortise after figuring out how to handle quarter collisions.

`generate_employee_results` and `generate_survival` are enormous and should be modularised.

Current method for writing results files is terrible, replace with command line arguments and write results to STDOUT.
