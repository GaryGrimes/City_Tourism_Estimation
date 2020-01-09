import requests
import json
import time
import math
import pandas as pd
import googlemaps
from key import get_key


def haversine(lon1, lat1, lon2, lat2):
    b = math.pi / 180
    c = math.sin((lat2 - lat1) * b / 2)
    d = math.sin((lon2 - lon1) * b / 2)
    a = c * c + d * d * math.cos(lat1 * b) * math.cos(lat2 * b)
    return 12756274 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def grid_cut(left_top, right_bottom, distance):
    # left_bottom:左下角坐标 right_top：右上角坐标
    # part_distance:切片的边长
    y_dis = haversine(left_top[0], left_top[1], left_top[0], right_bottom[1])  # 横向距离
    x_dis = haversine(left_top[0], left_top[1], right_bottom[0], left_top[1])  # 纵向距离
    x_n = int(x_dis / distance) + 1  # 横向切片个数
    y_n = int(y_dis / distance) + 1  # 纵向切片个数
    x_range = right_bottom[0] - left_top[0]  # 横向经度差
    y_range = left_top[1] - right_bottom[1]  # 纵向纬度差
    part_x = x_range / x_n  # 切片横向距离
    part_y = y_range / y_n  # 切片纵向距离
    part_list = []
    for i in range(x_n):
        for j in range(y_n):
            part_left_top = [left_top[0] + i * part_x, left_top[1] - j * part_y]
            part_right_bottom = [left_top[0] + (i + 1) * part_x, left_top[1] - (j + 1) * part_y]
            part_list.append([part_left_top, part_right_bottom])
    loc_list = []
    for part in part_list:
        center = [(part[0][0] + part[1][0]) / 2, (part[0][1] + part[1][1]) / 2]
        loc_list.append(center)
    return loc_list


class GooglePlaces(object):
    def __init__(self, apikey):
        super(GooglePlaces, self).__init__()
        self.apiKey = apikey

    def high_end_search(self, location, radius, types):
        endpoint_url = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
        places = []
        params = {
            'location': location,
            'radius': radius,
            'keyword': types,
            'minprice': 3,
            'maxprice': 4,
            'rank_by': 'distance',
            'key': self.apiKey
        }
        res = requests.get(endpoint_url, params=params)
        results = json.loads(res.content)
        places.extend(results['results'])
        time.sleep(2)
        while "next_page_token" in results:
            params['pagetoken'] = results['next_page_token'],
            res = requests.get(endpoint_url, params=params)
            results = json.loads(res.content)
            places.extend(results['results'])
            time.sleep(2)
        return places

    def search_places_by_coordinate(self, location, radius, types):
        endpoint_url = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
        places = []
        params = {
            'location': location,
            'radius': radius,
            'keyword': types,
            'key': self.apiKey
        }
        res = requests.get(endpoint_url, params=params)
        results = json.loads(res.content)
        places.extend(results['results'])
        time.sleep(2)
        while "next_page_token" in results:
            params['pagetoken'] = results['next_page_token'],
            res = requests.get(endpoint_url, params=params)
            results = json.loads(res.content)
            places.extend(results['results'])
            time.sleep(2)
        return places

    def get_place_details(self, place_id, fields):
        endpoint_url = "https://maps.googleapis.com/maps/api/place/details/json"
        params = {
            'placeid': place_id,
            'fields': ",".join(fields),
            'key': self.apiKey
        }
        res = requests.get(endpoint_url, params=params)
        place_details = json.loads(res.content)
        return place_details


def bound_check(left_top, right_bottom):
    lng1, lat1 = left_top[0], left_top[1]
    lng2, lat2 = right_bottom[0], right_bottom[1]
    if lng1 < lng2 and lat1 > lat2:
        width = haversine(lng1, lat1, lng2, lat1)
        tall = haversine(lng1, lat1, lng1, lat2)
        print('Area size: %dm (height) * %dm (width)' % (tall, width))
        return 0
    else:
        print('Area boundary error!')
        return 1


API_key = get_key()
# approach 1: using local google_places script for querying
api = GooglePlaces(API_key)
# approach 2: using Git google_maps scripts
gmaps = googlemaps.Client(key=API_key)

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
lats, lngs = [], []
for i in range(1, 37):
    lats.append(boundaries[i][0][1])
    lats.append(boundaries[i][1][1])
    lngs.append(boundaries[i][0][0])
    lngs.append(boundaries[i][1][0])

min_lat, min_lng, max_lat, max_lng = min(lats), min(lngs), max(lats), max(lngs)

# TODO find place using location bias


flag = 1

if __name__ == '__main__' and flag:
    # boundary check for each area
    for name in areas:
        print('Checking area %d: ' % name)
        # skip error boundary areas
        bounds = {'left_top': boundaries[name][0], 'right_bottom': boundaries[name][1]}
        error_flag = bound_check(bounds['left_top'], bounds['right_bottom'])
        if error_flag == 1:
            continue
        # define initial search radius
        radius = 600
        part_distance = radius / (math.sqrt(2) / 2)  # unit: meters
        # query nodes
        nodes = grid_cut(bounds['left_top'], bounds['right_bottom'], part_distance)
        # adjust search radius such that the number of query nodes are fewer than 10
        while len(nodes) > 10:
            radius += 100
            part_distance = radius / (math.sqrt(2) / 2)  # unit: meters
            # query nodes
            nodes = grid_cut(bounds['left_top'], bounds['right_bottom'], part_distance)
        print('Search radius: %d and required query nodes: %d \n' % (radius, len(nodes)))
        # create object from class and begin search
        print('Distance to boundary corner: %dm' % haversine(nodes[0][0], nodes[0][1], bounds['left_top'][0],
                                                             bounds['left_top'][1]))
        try:
            print(
                'Distance between query nodes(centroids): %dm' % haversine(nodes[0][0], nodes[0][1], nodes[1][0],
                                                                           nodes[1][1]))
        except IndexError:
            pass

        # place search
        places_res = []
        # search restaurants
        types = ['food']
        if not 0 < radius:
            exit('radius error')
        for node in nodes:
            node = [node[1], node[0]]  # bounds输入的时候是反的
            for type in types:
                print('Parsing %ss around node: %s, distance as %d' % (type, ','.join([str(x) for x in node]),
                                                                       radius))
                place = api.high_end_search(','.join([str(x) for x in node]), str(radius), type)
                print('Found %d %s(s) around node' % (len(place), type))
                places_res.extend(place)
        # TODO print输出内容简化:
        #  把distance改为radius; around node 的坐标改为index

        # save results in pd
        Data_wide = pd.DataFrame(
            columns=['place_id', 'id', 'name', 'lat', 'lng', 'rating', 'user_ratings_total', 'vicinity', 'types'])
        for i in range(len(places_res)):
            lat, lng = places_res[i]['geometry']['location']['lat'], places_res[i]['geometry']['location']['lng']
            Data_wide.loc[i, 'place_id':'vicinity'] = places_res[i]['place_id'], places_res[i]['id'], places_res[i][
                'name'], lat, lng, places_res[i]['rating'], places_res[i]['user_ratings_total'], places_res[i][
                                                          'vicinity']
            Data_wide.at[i, 'types'] = places_res[i]['types']

        # remove duplicate results
        Data_wide.drop_duplicates(subset='place_id', inplace=True)
        print('Number of POIs found around area %d: %d' % (name, len(Data_wide)))
        Data_wide.to_csv('./Gourmet and leisure/' + str(name) + '_wide.csv', index=False)

        # remove POIs out of boundary
        Data = pd.DataFrame(
            columns=['place_id', 'id', 'name', 'lat', 'lng', 'rating', 'user_ratings_total', 'vicinity', 'types'])
        for i in range(len(places_res)):
            # boundary check
            lat, lng = places_res[i]['geometry']['location']['lat'], places_res[i]['geometry']['location']['lng']
            if bounds['left_top'][0] < lng < bounds['right_bottom'][0] and bounds['left_top'][1] > lat > \
                    bounds['right_bottom'][1]:
                Data.loc[i, 'place_id':'vicinity'] = places_res[i]['place_id'], places_res[i]['id'], places_res[i][
                    'name'], \
                                                     lat, lng, places_res[i]['rating'], places_res[i][
                                                         'user_ratings_total'], \
                                                     places_res[i]['vicinity']
                Data.at[i, 'types'] = places_res[i]['types']
        # remove duplicate results
        Data.drop_duplicates(subset='place_id', inplace=True)
        print('Number of POIs found inside area boundary %d: %d' % (name, len(Data)))
        # write results into folder: 'Temple and shrine'
        Data.to_csv('./Gourmet and leisure/' + str(name) + '.csv', index=False)
