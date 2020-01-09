# importing googlemaps module
import pandas as pd
import numpy as np
import time
import os
from datetime import datetime, date, timedelta


def create_latlng(node):
    lat, lng = node[1], node[0]
    return {'lat': lat, 'lng': lng}


def calculate_cycle_time(x):
    if x > 100000:
        return x
    else:
        return x // 3


# specify to-do works
flag = 1

# cycling information is not supported by either direction or distance matrix API
# transit data re-processing
if __name__ == '__main__' and flag:
    mode = 'walking'
    # time
    filename = 'travel_time_matrix_' + mode + '.csv'
    temp = pd.read_csv(os.path.join(os.path.dirname(__file__), 'Database', 'Final', filename), index_col=0)
    for col in temp.columns:
        temp[col] = temp[col].apply(lambda x: calculate_cycle_time(x))
    name_tempt = '_'.join(['travel_time_matrix', 'bicycling']) + '.csv'
    temp.to_csv(os.path.join(os.path.dirname(__file__), 'Database', name_tempt))