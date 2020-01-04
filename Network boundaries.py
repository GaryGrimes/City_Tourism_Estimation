""" This script defines area boundaries and calculates area centers."""
import pandas as pd
import numpy as np
import math
import os
from datetime import datetime, date, timedelta


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


def haver_dist(lon1, lat1, lon2, lat2):
    """Input: geological coordinates of two locations, in a list [lon, lat]. Output:
    Eculidian distance between the two locations in meters."""

    b = math.pi / 180
    c = math.sin((lat2 - lat1) * b / 2)
    d = math.sin((lon2 - lon1) * b / 2)
    a = c * c + d * d * math.cos(lat1 * b) * math.cos(lat2 * b)
    return 12756274 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


if __name__ == '__main__':

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

    for _ in boundaries:
        centers[_] = ((np.array(boundaries[_][0]) + np.array(boundaries[_][1])) / 2).tolist()

    origins = destinations = centers

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
    # 34.9963851, 135.7594209
    centers[39] = centers[38] = [34.9963851, 135.7594209]

    node_list[39].center = node_list[40].center = '京都駅'
    centers[41] = centers[40] = [34.9853387, 135.7583234]

    node_list[41].center = '西大路駅'
    centers[42] = [34.98115, 135.73208]

    node_list[42].center = '河原町駅'
    centers[43] = [35.00372, 135.76882]

    node_list[43].center = '三条駅'
    centers[44] = [35.00886, 135.77217]

    node_list[44].center = {'lat': 34.945958, 'lng': 135.761225}
    centers[45] = [34.945958, 135.761225]
    node_list[45].center = ['宝ヶ池駅', '嵐電嵯峨']
    centers[46] = [[35.0581761, 135.7923442], [35.0164451, 135.6814425]]

    node_list[46].center = '烏丸御池駅'
    centers[47] = [35.0108650, 135.7596385]

    # %%
    # todo: check haversine func. outputs distance in km or m?
    centers.items()

    # %% test haversine distance
    coord_home = [30.3279631, 120.1381562]
    coord_home.reverse()
    coord_campus = [30.3107963, 120.0820759]
    coord_campus.reverse()
    dist_test = haver_dist(*coord_home, *coord_campus)
