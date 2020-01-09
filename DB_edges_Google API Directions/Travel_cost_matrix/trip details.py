# importing googlemaps module
import googlemaps
import pandas as pd
import numpy as np
import time
import os
from datetime import datetime, date, timedelta
from key import get_key


def create_latlng(node):
    lat, lng = node[1], node[0]
    return {'lat': lat, 'lng': lng}


# specify to-do works
query_flag, clean_flag = 0, 0
transit_flag, re_inquery_flag = 0, 0

API_key = get_key()

# query data

# if __name__ == '__main__' and query_flag:
#     # Requires API key
#     gmaps = googlemaps.Client(key=API_key)
#
#     # create origin/destination list
#     areas = range(1, 38)
#     boundaries = {1: [[135.8074, 35.1265], [135.8418, 35.0635]],
#                   2: [[135.753141, 35.126590], [135.775803, 35.104627]],
#                   3: [[135.765895, 35.083676], [135.790349, 35.056074]],
#                   4: [[135.7478, 35.0631], [135.7602, 35.0529]],
#                   5: [[135.6661, 35.0624], [135.6839, 35.0519]],
#                   6: [[135.7922, 35.0561], [135.8099, 35.0430]],
#                   7: [[135.726436, 35.056776], [135.734913, 35.050022]],
#                   8: [[135.7569, 35.0536], [135.7705, 35.0443]],
#                   9: [[135.7421, 35.0471], [135.7522, 35.0404]],
#                   10: [[135.7257, 35.0422], [135.7333, 35.0369]],
#                   11: [[135.7670, 35.0412], [135.7785, 35.0289]],
#                   12: [[135.7306, 35.0352], [135.7429, 35.0269]],
#                   13: [[135.7092, 35.0365], [135.7284, 35.0274]],
#                   14: [[135.6625, 35.0306], [135.6924, 35.0191]],
#                   15: [[135.79714, 35.02816], [135.80012, 35.02570]],
#                   16: [[135.7899, 35.0254], [135.7985, 35.0095]],
#                   17: [[135.7778, 35.0233], [135.7901, 35.0105]],
#                   18: [[135.7591, 35.0345], [135.7675, 35.0175]],
#                   19: [[135.7156, 35.0259], [135.7302, 35.0189]],
#                   20: [[135.7426, 35.0173], [135.7552, 35.0113]],
#                   21: [[135.7362, 35.0144], [135.7426, 35.0085]],
#                   22: [[135.7043, 35.0175], [135.7142, 35.0132]],
#                   23: [[135.6651, 35.0198], [135.6832, 35.0095]],
#                   24: [[135.7724, 35.0092], [135.7863, 34.9983]],
#                   25: [[135.7599, 35.0105], [135.7713, 34.9997]],
#                   26: [[135.6791, 35.0019], [135.6900, 34.9901]],
#                   27: [[135.7755, 34.9967], [135.7857, 34.9920]],
#                   28: [[135.7655, 34.9923], [135.7765, 34.9860]],
#                   29: [[135.7461, 34.9956], [135.7635, 34.9830]],
#                   # the area was extended to include the Kyoto station bldg (shopping and gourmet)
#                   30: [[135.7067, 34.9863], [135.7138, 34.9808]],
#                   31: [[135.7694, 34.9849], [135.7822, 34.9748]],
#                   32: [[135.7424, 34.9842], [135.7506, 34.9779]],
#                   33: [[135.7688, 34.9723], [135.7837, 34.9627]],
#                   34: [[135.8163, 34.9538], [135.8250, 34.9484]],
#                   35: [[135.74553, 34.95167], [135.74933, 34.94878]],
#                   36: [[135.7534, 34.9337], [135.7626, 34.9289]],
#                   37: [[135.6076, 35.2130], [135.7253, 35.1223]]
#                   }
#     centers = {}
#     daytimes = '7:30 12:30 17:30'.split()
#
#     modes = 'driving, walking, transit'.split(', ')
#
#     for foo in boundaries:
#         centers[foo] = ((np.array(boundaries[foo][0]) + np.array(boundaries[foo][1])) / 2).tolist()
#
#     origins = centers
#     destinations = centers
#
#     # origin and destination are lists of nodes
#     origin, destination = [], []
#     # create 'origin' and 'destination' for func. input
#     for i in origins:
#         try:
#             if isinstance(origins[i], list):
#                 origin.append(create_latlng(origins[i]))
#             elif isinstance(origins[i], str):
#                 origin.append(origins[i])
#             else:
#                 print('Value error, break.')
#                 break
#         except KeyError:
#             print('origins input error!')
#
#     for i in destinations:
#         try:
#             if isinstance(destinations[i], list):
#                 destination.append(create_latlng(destinations[i]))
#             elif isinstance(destinations[i], str):
#                 destination.append(destinations[i])
#             else:
#                 print('Value error, break.')
#                 break
#         except KeyError:
#             print('origins input error!')
#
#     # in our case, origin == destination
#     origin = destination
#     replace_flag = 1
#     while replace_flag == 1:
#         origin[0] = 'Yamashiro Ohara Post Office, 246 Ōhararaikōinchō, Sakyō-ku, Kyoto, Kyoto'
#         origin[1] = 'Kibuneguchi Station, Kuramakibunecho, 左京区京都市京都府'  # {'lat': 35.105941, 'lng': 135.763549}
#         origin[2] = '岩仓（京都）'  # {'lat': 35.071065, 'lng': 135.786855}
#         origin[3] = 'Kamigamo-jinja, 339 Kamigamo Motoyama, 北区京都市京都府'
#         origin[5] = 'Manshu-in Monzeki Temple'
#         origin[6] = 'Zuiho Temple'  # '{'lat': 35.054116, 'lng': 135.732038}  # 光悦寺
#         origin[9] = 'Kitayama Rokuon Temple'
#         origin[13] = '京都府京都市右京区 Sagashakadofujinokicho, 46清凉寺'
#         origin[14] = '京都府京都市左京区 Ginkakujicho, 2慈照寺'
#         origin[19] = '二条城前'
#         origin[22] = {'lat': 35.013654, 'lng': 135.677851}
#         origin[25] = '松尾大社'
#         origin[26] = '清水寺'
#         origin[28] = 'ヨドバシカメラ マルチメディア京都'
#         origin[29] = {'lat': 34.984515, 'lng': 135.706932}  # {'lat': 34.982897, 'lng': 135.710993}
#         origin[32] = '京都府京都市伏见区 Fukakusa Yabunouchicho, 68 伏見稲荷大社'
#         origin[36] = {'lat': 35.179692, 'lng': 135.659977}
#         replace_flag = 0
#
#     # creating two empty dicts to store time and distance 'Panels' for each time period
#     P_time, P_distance = {}, {}
#     for daytime in daytimes:
#         # create query time
#         temp = (date.today() + timedelta(days=1)).strftime("%Y-%m-%d ") + daytime + ':00'
#         d_time = datetime.strptime(temp, "%Y-%m-%d %H:%M:%S")
#         size = [len(origin), len(destination)]
#         # store travel time data for multiple modes
#         Data_time, Data_distance = {}, {}
#
#         for mode in modes:
#             print('Requiring results for mode %s at time period %s.' % (mode, daytime))
#             travel_distance, travel_time, traffic_time, travel_cost = pd.DataFrame(np.zeros(size)), pd.DataFrame(
#                 np.zeros(size)), pd.DataFrame(np.zeros(size)), pd.DataFrame(np.zeros(size))
#             element_count = 0
#             for i in range(len(origin) - 1):  # 上三角矩阵，不能到最后一行
#                 print('Query from origin %d:' % (i + 1))
#                 # sleep 10 sec if query element number exceeds the limit
#                 query_count = len(range(i + 1, size[1]))
#                 print('Current query destinations: %d ~ %d, query counts: %d' % ((i + 2), size[1], query_count))
#                 if element_count + query_count > 100:
#                     print('Query count exceeds server limits. Sleep for 10 secs.')
#                     time.sleep(10)
#                     element_count = 0
#                     print('Waking up and continuing...')
#                 element_count += query_count
#                 # query result
#                 if mode == 'walking':
#                     details = gmaps.distance_matrix(origin[i], destination[i + 1:size[1]], mode=mode)
#                 if mode == 'driving':
#                     # include traffic conditions
#                     details = gmaps.distance_matrix(origin[i], destination[i + 1:size[1]], mode=mode,
#                                                     departure_time=d_time, traffic_model='best_guess')
#                 else:
#                     details = gmaps.distance_matrix(origin[i], destination[i + 1:size[1]], mode=mode,
#                                                     departure_time=d_time)
#                 # 填充DataFrame啦！
#                 query_result = details['rows'][0]['elements']
#
#                 ttime, tdist = [], []
#                 for x in query_result:
#                     try:
#                         ttime.append(x['duration']['value'])
#                         tdist.append(x['distance']['value'])
#                     except KeyError:
#                         print('No duration or distance found for mode %s. Replaced with infinity.' % mode)
#                         ttime.append(999999)
#                         tdist.append(999999)
#
#                 # time 和 distance是必须的
#                 travel_distance.at[i, i + 1:size[1]], travel_time.at[i, i + 1:size[1]] = tdist, ttime
#
#                 # traffic duration for mode 'driving'
#                 if mode == 'driving':
#                     ttraffic = []
#                     for x in query_result:
#                         try:
#                             ttraffic.append(x['duration_in_traffic']['value'])
#                         except KeyError:
#                             print('Cannot find duration in traffic for route. ')  # TODO 添加报错为route from a to b
#                             ttraffic.append(x['duration']['value'])
#                     # driving time considering traffic (optimistic/pessimistic/best guess
#                     traffic_time.at[i, i + 1:size[1]] = ttraffic
#
#                 # transit cost for mode 'transit'
#                 if mode == 'transit':
#                     tcost = []
#                     for x in query_result:
#                         try:
#                             tcost.append(x['fare']['value'])
#                         except KeyError:
#                             print('Cannot find transit fare for route.')
#                             tcost.append(999999)
#                     travel_cost.at[i, i + 1:size[1]] = tcost
#
#             # 填充剩余矩阵
#             for i in range(size[0]):
#                 for j in range(size[1]):
#                     if i > j:
#                         travel_time.loc[i, j] = travel_time.loc[j, i]
#                         travel_distance.loc[i, j] = travel_distance.loc[j, i]
#                         if mode == 'driving':
#                             traffic_time.loc[i, j] = traffic_time.loc[j, i]
#                         if mode == 'transit':
#                             travel_cost.loc[i, j] = travel_cost.loc[j, i]
#
#             # travel time output
#             name_tempt = daytime + '_' + mode + '_' + 'travel_time' + '.csv'
#             travel_time.to_csv(os.path.join(os.path.dirname(__file__), 'Database', name_tempt))
#             # travel distance output
#             name_tempt = daytime + '_' + mode + '_' + 'travel_distance' + '.csv'
#             travel_distance.to_csv(os.path.join(os.path.dirname(__file__), 'Database', name_tempt))
#             # traffic time
#             if mode == 'driving':
#                 name_tempt = daytime + '_' + mode + '_' + 'duration_in_traffic' + '.csv'
#                 traffic_time.to_csv(os.path.join(os.path.dirname(__file__), 'Database', name_tempt))
#                 pass
#             if mode == 'transit':
#                 name_tempt = daytime + '_' + mode + '_' + 'transit_fare' + '.csv'
#                 travel_cost.to_csv(os.path.join(os.path.dirname(__file__), 'Database', name_tempt))
#                 pass
#
#             Data_time[mode], Data_distance[mode] = travel_time, travel_distance
#         # P_time with capital character P is to store each dict at every time period in a dictionary
#         P_time[daytime], P_distance[daytime] = Data_time, Data_distance
#     print('\nQuery completed. Please check .csv data.')

# query data

# query using direction API
if __name__ == '__main__' and query_flag:
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

    for daytime in daytimes:
        # create query time
        temp = (date.today() + timedelta(days=1)).strftime("%Y-%m-%d ") + daytime + ':00'
        d_time = datetime.strptime(temp, "%Y-%m-%d %H:%M:%S")
        size = [len(origin), len(destination)]
        # store travel time data for multiple modes
        Data_time, Data_distance = {}, {}

        for mode in modes:
            print('Requiring results for mode %s at time period %s.' % (mode, daytime))
            travel_distance, travel_time, travel_cost = pd.DataFrame(np.zeros(size)), pd.DataFrame(
                np.zeros(size)), pd.DataFrame(np.zeros(size))

            element_count = 0
            for i in range(size[0]):
                for j in range(size[1]):
                    if i < j:  # only query the upper-triangular area of the matrix
                        # query
                        if j == i + 1:
                            print('\n ------ Query route data from origin %d for mode %s ------' % (i + 1, mode))
                        if mode == 'walking':
                            details = gmaps.directions(origin[i], destination[j], mode=mode)
                        if mode == 'driving':
                            # include traffic conditions
                            details = gmaps.directions(origin[i], destination[j], mode=mode,
                                                       departure_time=d_time, traffic_model='best_guess')
                        else:
                            details = gmaps.directions(origin[i], destination[j], mode=mode, arrival_time=d_time)

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
                            print('No duration or distance found for mode %s from %d to %d. Replaced with infinity.'
                                  % (mode, i + 1, j + 1))
                            ttime = tdist = 999999

                        # write information to DataFrames
                        travel_distance.loc[j, i] = travel_distance.loc[i, j] = tdist
                        travel_time.loc[j, i] = travel_time.loc[i, j] = ttime

                        # traffic duration for mode 'driving'
                        if mode == 'driving':
                            try:
                                pass
                            except KeyError:
                                print('Cannot find duration in traffic for route. ')  # TODO 添加报错为route from a to b

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

                            travel_cost.loc[j, i] = travel_cost.loc[i, j] = tcost

            # travel time output
            name_tempt = daytime + '_' + mode + '_' + 'travel_time' + '.csv'
            travel_time = travel_time.astype(int)
            travel_time.to_csv(os.path.join(os.path.dirname(__file__), 'Database', name_tempt))
            # travel distance output
            name_tempt = daytime + '_' + mode + '_' + 'travel_distance' + '.csv'
            travel_distance = travel_distance.astype(int)
            travel_distance.to_csv(os.path.join(os.path.dirname(__file__), 'Database', name_tempt))
            # traffic time
            if mode == 'driving':
                # any addition information to output?
                pass
            if mode == 'transit':
                name_tempt = daytime + '_' + mode + '_' + 'transit_fare' + '.csv'
                travel_cost = travel_cost.astype(int)
                travel_cost.to_csv(os.path.join(os.path.dirname(__file__), 'Database', name_tempt))
                pass
            Data_time[mode], Data_distance[mode] = travel_time, travel_distance
        # P_time with capital character P is to store each dict at every time period in a dictionary
    print('\nQuery completed. Please check .csv data.')

# data processing
if __name__ == '__main__' and clean_flag:
    daytimes = '7:30 12:30 17:30'.split()
    modes = 'driving, walking, transit'.split(', ')

    for mode in modes:
        # TODO 先汇总计算平均时间段的travel time, distance
        tt_mat, td_mat = pd.DataFrame(np.zeros([37, 37])), pd.DataFrame(np.zeros([37, 37]))
        if mode == 'transit':
            tf_mat = pd.DataFrame(np.zeros([37, 37]))
        if mode == 'driving':
            tdr_mat = pd.DataFrame(np.zeros([37, 37]))
        for daytime in daytimes:
            # travel_time
            name_temp = [daytime, mode, 'travel_time.csv']
            filename = '_'.join(name_temp)
            temp = pd.read_csv(os.path.join(os.path.dirname(__file__), 'Database', 'Backup', filename))
            temp.drop(columns=temp.columns[0], inplace=True)
            temp.columns = tt_mat.columns
            tt_mat += temp
            # travel_distance
            name_temp = [daytime, mode, 'travel_distance.csv']
            filename = '_'.join(name_temp)
            temp = pd.read_csv(os.path.join(os.path.dirname(__file__), 'Database', 'Backup', filename))
            temp.drop(columns=temp.columns[0], inplace=True)
            temp.columns = td_mat.columns
            td_mat += temp

            # TODO transit额外有cost， driving 额外有 duration in traffic
            if mode == 'transit':
                # write csv
                name_temp = [daytime, mode, 'transit_fare.csv']
                filename = '_'.join(name_temp)
                temp = pd.read_csv(os.path.join(os.path.dirname(__file__), 'Database', 'Backup', filename))
                temp.drop(columns=temp.columns[0], inplace=True)
                temp.columns = tf_mat.columns
                tf_mat += temp
                pass

            if mode == 'driving':
                # write csv
                name_temp = [daytime, mode, 'duration_in_traffic.csv']
                filename = '_'.join(name_temp)
                temp = pd.read_csv(os.path.join(os.path.dirname(__file__), 'Database', 'Backup', filename))
                temp.drop(columns=temp.columns[0], inplace=True)
                temp.columns = tdr_mat.columns
                # recalculate 因为之前计算duration in traffic的时候下三角补全算错了
                for i in range(temp.shape[0]):
                    for j in range(temp.shape[1]):
                        if i > j:
                            temp.loc[i, j] = temp.loc[j, i]
                temp.to_csv(os.path.join(os.path.dirname(__file__), 'Database', 'Backup', filename))
                tdr_mat += temp
                pass
        # output
        tt_mat = tt_mat[tt_mat.columns] / len(daytimes)
        name_tempt = '_'.join(['travel_time_matrix', mode]) + '.csv'
        tt_mat = tt_mat.astype(int)
        tt_mat.to_csv(os.path.join(os.path.dirname(__file__), 'Database', name_tempt))

        td_mat = td_mat[td_mat.columns] / len(daytimes)
        name_tempt = '_'.join(['travel_distance_matrix', mode]) + '.csv'
        td_mat = td_mat.astype(int)
        td_mat.to_csv(os.path.join(os.path.dirname(__file__), 'Database', name_tempt))

        if mode == 'transit':
            tf_mat = tf_mat[tf_mat.columns] / len(daytimes)
            name_tempt = '_'.join(['transit_fare_matrix']) + '.csv'
            tf_mat = tf_mat.astype(int)  # to int for better visualization
            tf_mat.to_csv(os.path.join(os.path.dirname(__file__), 'Database', name_tempt))

        if mode == 'driving':
            tdr_mat = tdr_mat[tdr_mat.columns] / len(daytimes)
            name_tempt = '_'.join(['duration_in_traffic_matrix', mode]) + '.csv'
            tdr_mat = tdr_mat.astype(int)
            tdr_mat.to_csv(os.path.join(os.path.dirname(__file__), 'Database', name_tempt))

# transit data re-processing
if __name__ == '__main__' and transit_flag:
    daytimes = '7:30 12:30 17:30'.split()
    modes = ['transit']

    for mode in modes:
        # TODO 先汇总计算平均时间段的travel time, distance
        tt_mat, td_mat = pd.DataFrame(np.zeros([37, 37])), pd.DataFrame(np.zeros([37, 37]))
        tf_mat = pd.DataFrame(np.zeros([37, 37]))
        # store trip info
        tt_dict, td_dict, tf_dict = {}, {}, {}
        for daytime in daytimes:
            # time
            name_temp = [daytime, mode, 'travel_time.csv']
            filename = '_'.join(name_temp)
            temp = pd.read_csv(os.path.join(os.path.dirname(__file__), 'Database', 'Backup', filename))
            temp.drop(columns=temp.columns[0], inplace=True)
            temp.columns = tt_mat.columns
            temp = temp.astype(int)
            tt_dict[daytime] = temp
            # distance
            name_temp = [daytime, mode, 'travel_distance.csv']
            filename = '_'.join(name_temp)
            temp = pd.read_csv(os.path.join(os.path.dirname(__file__), 'Database', 'Backup', filename))
            temp.drop(columns=temp.columns[0], inplace=True)
            temp.columns = tt_mat.columns
            temp = temp.astype(int)
            td_dict[daytime] = temp
            # fare
            name_temp = [daytime, mode, 'transit_fare.csv']
            filename = '_'.join(name_temp)
            temp = pd.read_csv(os.path.join(os.path.dirname(__file__), 'Database', 'Backup', filename))
            temp.drop(columns=temp.columns[0], inplace=True)
            temp.columns = tt_mat.columns
            temp = temp.astype(int)
            tf_dict[daytime] = temp

        for i in range(tt_mat.shape[0]):
            for j in range(tt_mat.shape[1]):
                if j > i:
                    # travel_time
                    weight_count, value = 0, 0
                    for daytime in daytimes:
                        res = tt_dict[daytime].loc[i, j]
                        if not res == 999999:
                            weight_count += 1
                            value += res
                    tt_mat.loc[j, i] = tt_mat.loc[i, j] = value / weight_count if weight_count else 999999

                    # travel_distance
                    weight_count, value = 0, 0
                    for daytime in daytimes:
                        res = td_dict[daytime].loc[i, j]
                        if not res == 999999:
                            weight_count += 1
                            value += res
                    td_mat.loc[j, i] = td_mat.loc[i, j] = value / weight_count if weight_count else 999999

                    # fare
                    weight_count, value = 0, 0
                    for daytime in daytimes:
                        res = tf_dict[daytime].loc[i, j]
                        if not res == 999999:
                            weight_count += 1
                            value += res
                    tf_mat.loc[j, i] = tf_mat.loc[i, j] = value / weight_count if weight_count else 999999

        # output
        name_tempt = '_'.join(['travel_time_matrix', mode]) + '.csv'
        tt_mat = tt_mat.astype(int)
        tt_mat.to_csv(os.path.join(os.path.dirname(__file__), 'Database', name_tempt))

        name_tempt = '_'.join(['travel_distance_matrix', mode]) + '.csv'
        td_mat = td_mat.astype(int)
        td_mat.to_csv(os.path.join(os.path.dirname(__file__), 'Database', name_tempt))

        if mode == 'transit':
            name_tempt = '_'.join(['transit_fare_matrix']) + '.csv'
            tf_mat = tf_mat.astype(int)
            tf_mat.to_csv(os.path.join(os.path.dirname(__file__), 'Database', name_tempt))

# transit data re-inquery
if __name__ == '__main__' and re_inquery_flag:
    gmaps = googlemaps.Client(key=API_key)
    # create origin/destination list
    areas = range(1, 38)
    daytimes = '7:30 12:30 17:30'.split()
    modes = 'driving, walking, transit'.split(', ')
    mode = 'transit'
    # origin and destination are lists of nodes
    # in our case, origin == destination
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
        origin[29] = '桂離宮'  # {'lat': 34.984515, 'lng': 135.706932}  # {'lat': 34.982897, 'lng': 135.710993}
        origin[32] = '京都府京都市伏见区 Fukakusa Yabunouchicho, 68 伏見稲荷大社'
        origin[36] = 'Jōshōkōji Temple, Maruyama-14-6 Keihokuidocho, Ukyo Ward, Kyoto, 601-0313'
        # {'lat': 35.179692, 'lng': 135.659977}
        replace_flag = 0

    # query for data

    # create query time
    # temp = (date.today() + timedelta(days=1)).strftime("%Y-%m-%d ") + daytime + ':00'
    # d_time = datetime.strptime(temp, "%Y-%m-%d %H:%M:%S")
    size = [len(origin), len(destination)]
    # store travel time data for multiple modes
    Data_time, Data_distance = {}, {}
    # print('Requiring results for mode %s at time period %s.' % ('transit', daytime))
    travel_distance, travel_time, travel_cost = pd.DataFrame(np.zeros(size)), pd.DataFrame(
        np.zeros(size)), pd.DataFrame(np.zeros(size))

    element_count = 0

    for i in range(len(origin) - 1):  # 上三角矩阵，不能到最后一行
        print('Query from origin %d:' % (i + 1))
        # sleep 10 sec if query element number exceeds the limit
        query_count = len(range(i + 1, size[1]))
        print('Current query destinations: %d ~ %d, query counts: %d' % ((i + 2), size[1], query_count))
        if element_count + query_count > 100:
            print('Query count exceeds server limits. Sleep for 10 secs.')
            time.sleep(10)
            element_count = 0
            print('Waking up and continuing...')
        element_count += query_count
        # query result (departure time not specified)
        details = gmaps.distance_matrix(origin[i], destination[i + 1:size[1]], mode='transit')
        # 填充DataFrame啦！
        query_result = details['rows'][0]['elements']

        ttime, tdist = [], []
        for x in query_result:
            try:
                ttime.append(x['duration']['value'])
                tdist.append(x['distance']['value'])
            except KeyError:
                print('No duration or distance found for mode %s. Replaced with infinity.' % mode)
                ttime.append(999999)
                tdist.append(999999)

        # time 和 distance是必须的
        travel_distance.at[i, i + 1:size[1]], travel_time.at[i, i + 1:size[1]] = tdist, ttime

        # transit cost for mode 'transit'
        if mode == 'transit':
            tcost = []
            for x in query_result:
                try:
                    tcost.append(x['fare']['value'])
                except KeyError:
                    print('Cannot find transit fare for route.')
                    tcost.append(999999)
            travel_cost.at[i, i + 1:size[1]] = tcost

    # 填充剩余矩阵
    for i in range(size[0]):
        for j in range(size[1]):
            if i > j:
                travel_time.loc[i, j] = travel_time.loc[j, i]
                travel_distance.loc[i, j] = travel_distance.loc[j, i]
                if mode == 'transit':
                    travel_cost.loc[i, j] = travel_cost.loc[j, i]

    # travel time output
    name_tempt = mode + '_' + 'travel_time' + '.csv'
    travel_time.to_csv(os.path.join(os.path.dirname(__file__), 'Database', name_tempt))
    # travel distance output
    name_tempt = mode + '_' + 'travel_distance' + '.csv'
    travel_distance.to_csv(os.path.join(os.path.dirname(__file__), 'Database', name_tempt))
    # traffic time
    if mode == 'transit':
        name_tempt = mode + '_' + 'transit_fare' + '.csv'
        travel_cost.to_csv(os.path.join(os.path.dirname(__file__), 'Database', name_tempt))
        pass

# retry_29 = 1
# if retry_29 == 1:
#     time, distance, cost = {}, {}, {}
#     for j in range(len(destination)):
#         details = gmaps.directions(destination[j], origin[29], mode=mode, arrival_time=d_time)
#         try:
#             query_result = details[0]
#         except IndexError:
#             query_result = {}
#             print('Result not found for mode %s from %d to %d ' % (mode, j + 1, 30))
#
#         try:
#             ttime, tdist = query_result['legs'][0]['duration']['value'], \
#                            query_result['legs'][0]['distance']['value']
#         except:
#             print('No duration or distance found for mode %s from %d to %d. Replaced with infinity.'
#                   % (mode, j + 1, 30))
#             ttime = tdist = 999999
#
#         # write information to DataFrames
#         distance[j] = tdist
#         time[j] = ttime
#
#         # transit cost for mode 'transit'
#         if mode == 'transit':
#             try:
#                 tcost = query_result['fare']['value']
#             except KeyError:
#                 print('Cannot find transit fare for route from %d to %d.' % (j + 1, 30))
#                 try:
#                     legs = query_result['legs']
#                     for leg in legs:
#                         steps = leg['steps']
#                         print('Find %d segments in current route.' % len(steps))
#                         walking_count = 0
#                         for step in steps:
#                             if step['travel_mode'] == 'WALKING':
#                                 walking_count += 1
#                             print('Found one segment by %s.' % step['travel_mode'])
#                         if walking_count == len(steps):
#                             print('Current route is recommended on foot.')
#                         tcost = 0
#                 except KeyError:
#                     tcost = 999999
#                     pass
#
#             cost[j] = tcost

# refill lower triangle
infix = 'transit_'
filename = ['transit_fare', 'travel_time']

for name in filename:
    temp = infix + name + '.csv'
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), 'Database', 'Final', temp), index_col=0)
    for i in range(df.shape[0]):
        for j in range(df.shape[1]):
            if i > j:
                df.iloc[i, j] = df.iloc[j, i]
    df.to_csv((name + '.csv'))
