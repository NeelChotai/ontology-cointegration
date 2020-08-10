from main import *

if path.isfile(GRAPH_CACHE):
    graph = pop_cache(GRAPH_CACHE)
else:
    graph = populate()
    push_cache(GRAPH_CACHE, graph)

employee_dict = pairs_with_attributes(query(graph, employee_type.EMPLOYEE))
return_dict = {}

for interval in range(1, 6):
    employee_pairs = [x for x in list(
        employee_dict) if employee_dict[x] >= interval]
    return_dict[interval] = len(employee_pairs)

print(return_dict)
