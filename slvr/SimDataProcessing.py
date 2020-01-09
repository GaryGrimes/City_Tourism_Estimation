"This module generates necessary data for simulation, including network (node and edge) properties, and phi."

import pandas as pd
import numpy as np
import os, pickle


class Tourist(object):
    tourist_count = 0

    def __init__(self, index, uid):
        self.index, self.uid = index, uid
        self.trip_df, self.time_budget = None, 0
        self.path_pdt, self.path_obs = None, None
        self.preference = None
        Tourist.tourist_count += 1


def print_path(path_to_print):
    print(list(np.array(path_to_print) + 1))


# %% setting up nodes
node_num = 37  # Number of attractions. Origin and destination are excluded.

# Important: attraction utility was not sorted by index previously
# todo: consider replace util in 1-6-8 by (gourmet utiltity + red leave and shrines)

Intrinsic_utilities = pd.read_excel(os.path.join(os.path.dirname(__file__), 'Database', 'Intrinsic Utility.xlsx'),
                                    sheet_name='data')
utility_matrix = []
for _idx in range(Intrinsic_utilities.shape[0]):
    temp = np.around(list(Intrinsic_utilities.iloc[_idx, 2:5]), decimals=3)
    utility_matrix.append(temp)
utility_matrix = np.array(utility_matrix)

# time
Dwell_time = pd.read_excel(os.path.join(os.path.dirname(__file__), 'Database', 'Dwell time array.xlsx'),
                           index_col=0)

# replace missing values by population average
Dwell_time.loc[35, 'mean'] = Dwell_time['mean'][Dwell_time['mean'] != 5].mean()  # Attraction 35
dwell_vector = np.array(Dwell_time['mean'])

# %% edge property
Edge_time_matrix = pd.read_excel(
    os.path.join(os.path.dirname(__file__), 'Database', 'Trips', 'Final', 'transit_time_update2.xlsx'), index_col=0)

edge_time_matrix = np.array(Edge_time_matrix)

# Edge travel time update
# need several iterations to make sure direct travel is shorter than any detour

no_update, itr = 0, 0
# print('Starting travel_time_check...')
for _ in range(3):
    while not no_update:
        # print('Current iteration: {}'.format(itr + 1))
        no_update = 1
        for i in range(edge_time_matrix.shape[0] - 1):
            for j in range(i + 1, edge_time_matrix.shape[0]):
                time = edge_time_matrix[i, j]
                shortest_node, shortest_time = 0, time
                for k in range(edge_time_matrix.shape[0]):
                    if edge_time_matrix[i, k] + edge_time_matrix[k, j] < shortest_time:
                        shortest_node, shortest_time = k, edge_time_matrix[i, k] + edge_time_matrix[k, j]
                if shortest_time < time:
                    no_update = 0
                    # print('travel time error between {0} and {1}, \
                    # shortest path is {0}-{2}-{1}'.format(i, j, shortest_node))
                    edge_time_matrix[j, i] = edge_time_matrix[i, j] = shortest_time
        itr += 1
        if no_update:
            # print('Travel time update complete.\n')
            pass

# Edge travel cost (fare)
Edge_cost_matrix = pd.read_csv(
    os.path.join(os.path.dirname(__file__), 'Database', 'Trips', 'Final', 'wide_transit_fare_matrix.csv'),
    index_col=0)
# Edge travel distance
Edge_distance_matrix = pd.read_excel(
    os.path.join(os.path.dirname(__file__), 'Database', 'Trips', 'Final', 'driving_wide_distance_matrix.xlsx'),
    index_col=0)

edge_cost_matrix = np.array(Edge_cost_matrix)
# distance matrix for path penalty evaluation
edge_distance_matrix = np.array(Edge_distance_matrix)  # distance between attraction areas

#  check UtilMatrix等各个matrix的shape是否正确。与NodeNum相符合
if len(utility_matrix) != node_num:
    raise ValueError('Utility matrix error.')
if edge_time_matrix.shape[0] != edge_time_matrix.shape[1]:
    raise ValueError('Time matrix error.')
if edge_cost_matrix.shape[0] != edge_cost_matrix.shape[1]:
    raise ValueError('Cost matrix error.')
if len(dwell_vector) != node_num:
    raise ValueError('Dwell time array error.')
# %% setting up behavior parameters
phi = 0.1

# print('Simulation data processing complete.')
