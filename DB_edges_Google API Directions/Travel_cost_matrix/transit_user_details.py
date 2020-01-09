# %% importing googlemaps module
import googlemaps
import pandas as pd
import numpy as np
import os
from datetime import datetime, date, timedelta
from key import get_key


class Node(object):
    node_count = 0

    def __init__(self, idx):
        self.index = idx
        self.center = None
        Node.node_count += 1


class Edge(object):
    edge_count = 0

    def __init__(self, o, d):
        self.origin, self.destination = o, d
        self.transit_travel_time, self.transit_travel_distance, self.transit_fare = None, None, None
        Edge.edge_count += 1


def create_latlng(node):
    lat, lng = node[1], node[0]
    return {'lat': lat, 'lng': lng}


# specify to-do works
query_flag = 1

API_key = get_key()

if __name__ == '__main__' and query_flag:
    # read current time and cost matrix
    filename = 'travel_time_matrix_transit.csv'
    temp_time = pd.read_csv(os.path.join(os.path.dirname(__file__), 'Database', 'Final', filename), index_col=0)

    filename = 'fare_matrix_transit.csv'
    temp_cost = pd.read_csv(os.path.join(os.path.dirname(__file__), 'Database', 'Final', filename), index_col=0)
    size_temp = temp_time.shape
    # create time matrix and cost matrix
    size = [47, 47]
    Time_matrix, Cost_Matrix = pd.DataFrame(np.zeros(size), dtype=int), pd.DataFrame(np.zeros(size), dtype=int)
    Time_matrix.loc[0:36, 0:36] = np.array(temp_time)  # the index are in mess (changed to string)
    Cost_Matrix.loc[0:36, 0:36] = np.array(temp_cost)  # the index are in mess (changed to string)

    # Requires API key
    gmaps = googlemaps.Client(key=API_key)

    # create origin/destination list
    areas = range(1, 38)
    boundaries = {1: [[135.8074, 35.1265], [135.8418, 35.0635]],
                  2: [[135.753141, 35.126590], [135.775803, 35.104627]],
                  3: [[135.765895, 35.083676], [135.790349, 35.056074]],
                  4: [[135.7478, 35.0631], [135.7602, 35.0529]],
                  5: [[135.6661, 35.0624], [135.6839, 35.0519]],
                  6: [[135.7922, 35.0561], [135.8099, 35.0430]],
                  7: [[135.726436, 35.056776], [135.734913, 35.050022]],
                  8: [[135.7569, 35.0536], [135.7705, 35.0443]],
                  9: [[135.7421, 35.0471], [135.7522, 35.0404]],
                  10: [[135.7257, 35.0422], [135.7333, 35.0369]],
                  11: [[135.7670, 35.0412], [135.7785, 35.0289]],
                  12: [[135.7306, 35.0352], [135.7429, 35.0269]],
                  13: [[135.7092, 35.0365], [135.7284, 35.0274]],
                  14: [[135.6625, 35.0306], [135.6924, 35.0191]],
                  15: [[135.79714, 35.02816], [135.80012, 35.02570]],
                  16: [[135.7899, 35.0254], [135.7985, 35.0095]],
                  17: [[135.7778, 35.0233], [135.7901, 35.0105]],
                  18: [[135.7591, 35.0345], [135.7675, 35.0175]],
                  19: [[135.7156, 35.0259], [135.7302, 35.0189]],
                  20: [[135.7426, 35.0173], [135.7552, 35.0113]],
                  21: [[135.7362, 35.0144], [135.7426, 35.0085]],
                  22: [[135.7043, 35.0175], [135.7142, 35.0132]],
                  23: [[135.6651, 35.0198], [135.6832, 35.0095]],
                  24: [[135.7724, 35.0092], [135.7863, 34.9983]],
                  25: [[135.7599, 35.0105], [135.7713, 34.9997]],
                  26: [[135.6791, 35.0019], [135.6900, 34.9901]],
                  27: [[135.7755, 34.9967], [135.7857, 34.9920]],
                  28: [[135.7655, 34.9923], [135.7765, 34.9860]],
                  29: [[135.7461, 34.9956], [135.7635, 34.9830]],
                  # the area was extended to include the Kyoto station bldg (shopping and gourmet)
                  30: [[135.7067, 34.9863], [135.7138, 34.9808]],
                  31: [[135.7694, 34.9849], [135.7822, 34.9748]],
                  32: [[135.7424, 34.9842], [135.7506, 34.9779]],
                  33: [[135.7688, 34.9723], [135.7837, 34.9627]],
                  34: [[135.8163, 34.9538], [135.8250, 34.9484]],
                  35: [[135.74553, 34.95167], [135.74933, 34.94878]],
                  36: [[135.7534, 34.9337], [135.7626, 34.9289]],
                  37: [[135.6076, 35.2130], [135.7253, 35.1223]]
                  }
    centers = {}
    # daytimes = '7:30 12:30 17:30'.split()
    daytimes = ['12:30']

    # modes = 'driving, walking, transit'.split(', ')
    modes = ['transit']

    for foo in boundaries:
        centers[foo] = ((np.array(boundaries[foo][0]) + np.array(boundaries[foo][1])) / 2).tolist()

    origins = centers
    destinations = centers

    # origin and destination are lists of nodes
    origin, destination = [], []
    # create 'origin' and 'destination' for func. input
    for i in origins:
        try:
            if isinstance(origins[i], list):
                origin.append(create_latlng(origins[i]))
            elif isinstance(origins[i], str):
                origin.append(origins[i])
            else:
                print('Value error, break.')
                break
        except KeyError:
            print('origins input error!')

    for i in destinations:
        try:
            if isinstance(destinations[i], list):
                destination.append(create_latlng(destinations[i]))
            elif isinstance(destinations[i], str):
                destination.append(destinations[i])
            else:
                print('Value error, break.')
                break
        except KeyError:
            print('origins input error!')

    # in our case, origin == destination
    origin = destination
    replace_flag = 1
    while replace_flag == 1:
        origin[0] = 'Yamashiro Ohara Post Office, 246 Ōhararaikōinchō, Sakyō-ku, Kyoto, Kyoto'
        origin[1] = 'Kibuneguchi Station, Kuramakibunecho, 左京区京都市京都府'  # {'lat': 35.105941, 'lng': 135.763549}
        origin[2] = '岩仓（京都）'  # {'lat': 35.071065, 'lng': 135.786855}
        origin[3] = 'Kamigamo-jinja, 339 Kamigamo Motoyama, 北区京都市京都府'
        origin[5] = 'Manshu-in Monzeki Temple'
        origin[6] = 'Zuiho Temple'  # '{'lat': 35.054116, 'lng': 135.732038}  # 光悦寺
        origin[9] = 'Kitayama Rokuon Temple'
        origin[13] = '京都府京都市右京区 Sagashakadofujinokicho, 46清凉寺'
        origin[14] = '京都府京都市左京区 Ginkakujicho, 2慈照寺'
        origin[19] = '二条城前'
        origin[22] = {'lat': 35.013654, 'lng': 135.677851}
        origin[25] = '松尾大社'
        origin[26] = '清水寺'
        origin[28] = 'ヨドバシカメラ マルチメディア京都'
        origin[
            29] = 'Katsuraekihigashiguchi Bus Stop'  # {'lat': 34.982897, 'lng': 135.710993}
        origin[32] = '京都府京都市伏见区 Fukakusa Yabunouchicho, 68 伏見稲荷大社'
        origin[36] = 'Kyoto Municipal Shuzan Junior High School'
        replace_flag = 0

    # list of nodes, array of edges
    node_list, edge_list = [], []

    for count in range(47):
        # create node instances
        x = Node(count + 1)
        if x.index in origins:
            x.boundary = boundaries[x.index]
            x.center = origin[count]  # origin是array, 从0开始计数
        node_list.append(x)

    for slice in range(47):
        edge_slice = []
        for place in range(47):
            x = Edge(slice + 1, place + 1)
            edge_slice.append(x)
        edge_list.append(edge_slice)

    # %% query from origin: 37-46, destination: 0-46
    # fill centers for origin 37-46
    node_list[37].center = node_list[38].center = '烏丸五条'
    node_list[39].center = node_list[40].center = '京都駅'
    node_list[41].center = '西大路駅'
    node_list[42].center = '河原町駅'
    node_list[43].center = '三条駅'
    node_list[44].center = {'lat': 34.945958, 'lng': 135.761225}
    node_list[45].center = ['宝ヶ池駅', '嵐電嵯峨']
    node_list[46].center = '烏丸御池駅'
    # %% start query
    for daytime in daytimes:
        # create query time
        temp = (date.today() + timedelta(days=1)).strftime("%Y-%m-%d ") + daytime + ':00'
        d_time = datetime.strptime(temp, "%Y-%m-%d %H:%M:%S")

        # store travel time data for multiple modes

        for mode in modes:
            print('Requiring results for mode %s at time period %s.' % (mode, daytime))
            element_count = 0
            # for i in range(37, 47):
            for i in range(46, 47):
                for j in range(47):
                    if i > j:  # only query the upper-triangular area of the matrix
                        # query
                        if j == 0:
                            print('\n ------ Query route data from origin %d for mode %s ------' % (i + 1, mode))
                        # append every time (如果有多个center，选最小的一个作为time 和 fare)
                        travel_distance, travel_time, fare = [], [], []
                        # make them all into lists
                        _origins = node_list[i].center if type(node_list[i].center) == list else [node_list[i].center]
                        _destinations = node_list[j].center if type(node_list[j].center) == list else [
                            node_list[j].center]
                        # query start
                        for _origin in _origins:
                            for _destination in _destinations:
                                details = gmaps.directions(_origin, _destination, mode=mode, arrival_time=d_time)
                                # get time and distance information
                                try:
                                    query_result = details[0]
                                except IndexError:
                                    query_result = {}
                                    print('Result not found for mode %s from %d to %d ' % (mode, i + 1, j + 1))

                                try:
                                    ttime, tdist = query_result['legs'][0]['duration']['value'], \
                                                   query_result['legs'][0]['distance']['value']
                                except:
                                    print(
                                        'No duration or distance found for mode %s from %d to %d. Replaced with infinity.'
                                        % (mode, i + 1, j + 1))
                                    ttime = tdist = 999999
                                travel_time.append(ttime)
                                travel_distance.append(tdist)
                                # transit cost for mode 'transit'
                                if mode == 'transit':
                                    try:
                                        tcost = query_result['fare']['value']
                                    except KeyError:
                                        print('Cannot find transit fare for route from %d to %d.' % (i + 1, j + 1))
                                        try:
                                            legs = query_result['legs']
                                            for leg in legs:
                                                steps = leg['steps']
                                                print('Find %d segments in current route.' % len(steps))
                                                walking_count = 0
                                                for step in steps:
                                                    if step['travel_mode'] == 'WALKING':
                                                        walking_count += 1
                                                    print('Found one segment by %s.' % step['travel_mode'])
                                                if walking_count == len(steps):
                                                    print('Current route is recommended on foot.')
                                                tcost = 0
                                        except KeyError:
                                            tcost = 999999
                                            pass
                                fare.append(tcost)
                        edge_list[i][j].transit_travel_time = min(travel_time)
                        edge_list[i][j].transit_travel_distance = min(travel_distance)
                        edge_list[i][j].transit_fare = min(fare)

                        # write into df
                        Time_matrix.loc[i, j], Cost_Matrix.loc[i, j] = edge_list[i][j].transit_travel_time, \
                                                                       edge_list[i][j].transit_fare
            # %% post processing
            # 一些没结果的od，手动填补
            Time_matrix.loc[41, 0], Time_matrix.loc[38, 1] = 4200, 7899
            Cost_Matrix.loc[41, 0], Cost_Matrix.loc[38, 1] = 760, 980

            Time_matrix.loc[41, 4], Time_matrix.loc[41, 9] = 3840, 2880
            Cost_Matrix.loc[41, 4], Cost_Matrix.loc[41, 9] = 710, 460

            Time_matrix.loc[41, 36], Time_matrix.loc[41, 11] = 5880, 2160
            Cost_Matrix.loc[41, 36], Cost_Matrix.loc[41, 11] = 1410, 460

            Time_matrix.loc[41, 12], Time_matrix.loc[41, 18] = 2640, 2580
            Cost_Matrix.loc[41, 12], Cost_Matrix.loc[41, 18] = 460, 460

            Time_matrix.loc[41, 21], Time_matrix.loc[41, 25] = 1800, 2520
            Cost_Matrix.loc[41, 21], Cost_Matrix.loc[41, 25] = 450, 460

            Time_matrix.loc[44, 9], Time_matrix.loc[44, 36] = 2700, 7260
            Cost_Matrix.loc[44, 9], Cost_Matrix.loc[44, 36] = 670, 1580

            Time_matrix.loc[37:40, 29], Time_matrix.loc[42:46, 29] = [1620, 1620, 1740, 1740], \
                                                                     [1380, 1980, 2220, 2220, 1980]
            Cost_Matrix.loc[37:40, 29], Cost_Matrix.loc[42:46, 29] = [400, 400, 240, 240], [190, 190, 600, 380, 190]

            # 填补剩余矩阵
            for i in range(Time_matrix.shape[0]):
                for j in range(Time_matrix.shape[1]):
                    if i > j:
                        Time_matrix.loc[j, i] = Time_matrix.loc[i, j]
                        if mode == 'transit':
                            Cost_Matrix.loc[j, i] = Cost_Matrix.loc[i, j]

            # travel time output
            name_tempt = daytime + '_' + mode + '_' + 'time_matrix' + '.csv'
            Time_matrix.to_csv(os.path.join(os.path.dirname(__file__), 'Database', name_tempt))
            # transit fare output
            name_tempt = daytime + '_' + mode + '_' + 'fare_matrix' + '.csv'
            Cost_Matrix.to_csv(os.path.join(os.path.dirname(__file__), 'Database', name_tempt))

    print('\nQuery completed. Please check .csv data.')
