import requests
import json
import time
import math
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def haversine(lon1, lat1, lon2, lat2):
    b = math.pi / 180
    c = math.sin((lat2 - lat1) * b / 2)
    d = math.sin((lon2 - lon1) * b / 2)
    a = c * c + d * d * math.cos(lat1 * b) * math.cos(lat2 * b)
    return 12756274 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


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


def area_check(left_top, right_bottom, node_lat, node_lng):
    lng1, lat1 = left_top[0], left_top[1]
    lng2, lat2 = right_bottom[0], right_bottom[1]
    if lat1 > node_lat > lat2 and lng1 < node_lng < lng2:
        return 1
    else:
        return 0


def query(query_data):
    overpass_url = "http://overpass-api.de/api/interpreter"
    overpass_query = query_data if query_data else None
    print('Querying data...')
    response = requests.get(overpass_url,
                            params={'data': overpass_query})
    print('Query complete...')
    try:
        data = response.json()
        return data
    except Exception as e:
        print('str(e):\t\t', str(e))
        return None


def find_poi(query_line):
    data = query(query_line)
    results = {}
    out_region = 0
    # collect coords into list
    coords = []
    for x in data['elements']:
        if x['type'] == 'node':
            id, lat, lng = x['id'], x['lat'], x['lon']
            coords.append((lng, lat))
            try:
                tags = x['tags']
            except KeyError:
                # print('Current node %d is an element of a way or relation' % id)
                continue  # to the next element
        elif 'center' in x:
            id, lat, lng, tags = x['id'], x['center']['lat'], x['center']['lon'], x['tags']
            coords.append((lng, lat))
        else:
            print('Found other types: %s' % x['type'])
            break
        # distribute into areas
        for j in range(1, len(boundaries)):
            found_flag = 0
            if area_check(boundaries[j][0], boundaries[j][1], lat, lng):
                found_flag = 1
                if j not in results.keys():
                    results[j] = []
                results[j].append(x)
                break
        if not found_flag:
            out_region += 1
    # report
    print('\nAssignment results:')
    total = 0
    for i in sorted(results.keys()):
        total += len(results[i])
        print('Area %d: # %d' % (i, len(results[i])))
    print('Found total: # %d' % total)
    print('Out of region: # %d' % out_region)
    # plot
    # Convert coordinates into numpy array
    if coords:
        X = np.array(coords)
        plt.plot(X[:, 0], X[:, 1], 'o')
        plt.title('POI locations')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.axis('equal')
        plt.show()
    return results


# TODO query 都退化为类
class Poi:
    def __init__(self):
        self.places = {}


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

# example
example_query = """
[out:json];
area["ISO3166-1"="DE"][admin_level=2];
(node["amenity"="biergarten"](area);
 way["amenity"="biergarten"](area);
 rel["amenity"="biergarten"](area);
);
out center;
"""

pub_query = """
[out:json]
[timeout:60]
;
area(3600357794)->.searchArea;
(
  node
    ["amenity"="pub"]
    (area.searchArea);
  way
    ["amenity"="pub"]
    (area.searchArea);
  relation
    ["amenity"="pub"]
    (area.searchArea);
);
out center;
>;
out skel qt;
"""
bar_query = """[out:json]
[timeout:25]
;
area(3600357794)->.searchArea;
(
  node
    ["amenity"="bar"]
    (area.searchArea);
  way
    ["amenity"="bar"]
    (area.searchArea);
  relation
    ["amenity"="bar"]
    (area.searchArea);
);
out center;
>;
out skel qt;"""
restaurant_query = """[out:json]
[timeout:60]
;
area(3600357794)->.searchArea;
(
  node
    ["amenity"="restaurant"]
    (area.searchArea);
  way
    ["amenity"="restaurant"]
    (area.searchArea);
  relation
    ["amenity"="restaurant"]
    (area.searchArea);
);
out center;
>;
out skel qt;"""

# TODO query这些都改为函数
# for each query, results include places represented by nodes or ways (areas, buildings, etc.).
# For the latter case, nodes that the ways are made up of are not counted again...
# %% pubs
pubs = find_poi(pub_query)
# %% bars
bars = find_poi(bar_query)
# %% restaurants
restaurants = find_poi(restaurant_query)
# %% museum
museum_query = """[out:json]
[timeout:60]
;
area(3600357794)->.searchArea;
(
  node
    ["tourism"="museum"]
    (area.searchArea);
  way
    ["tourism"="museum"]
    (area.searchArea);
  relation
    ["tourism"="museum"]
    (area.searchArea);
);
out center;
>;
out skel qt;"""
museums = find_poi(museum_query)

cinema_query = """[out:json]
[timeout:60]
;
area(3600357794)->.searchArea;
(
  node
    ["amenity"="cinema"]
    (area.searchArea);
  way
    ["amenity"="cinema"]
    (area.searchArea);
  relation
    ["amenity"="cinema"]
    (area.searchArea);
);
out center;
>;
out skel qt;"""
cinemas = find_poi(cinema_query)

arts_query = """[out:json]
[timeout:60]
;
area(3600357794)->.searchArea;
(
  node
    ["amenity"="arts_centre"]
    (area.searchArea);
  way
    ["amenity"="arts_centre"]
    (area.searchArea);
  relation
    ["amenity"="arts_centre"]
    (area.searchArea);
);
out center;
>;
out skel qt;"""
art_centers = find_poi(arts_query)

# %% shops
shops_query = """[out:json]
[timeout:25]
;
area(3600357794)->.searchArea;
(
  node
    ["shop"]
    (area.searchArea);
  way
    ["shop"]
    (area.searchArea);
  relation
    ["shop"]
    (area.searchArea);
);
out center;
>;
out skel qt;"""
shops = find_poi(shops_query)
# %% output
columns = ['pub', 'bar', 'restaurant', 'museum', 'cinema', 'art_center', 'shop']
poi_table = pd.DataFrame(columns=columns, index=range(1, 38))
for i in poi_table.index:
    pub = len(pubs[i]) if i in pubs else 0
    bar = len(bars[i]) if i in bars else 0
    restaurant = len(restaurants[i]) if i in restaurants else 0
    museum = len(museums[i]) if i in museums else 0
    cinema = len(cinemas[i]) if i in cinemas else 0
    art_center = len(art_centers[i]) if i in art_centers else 0
    shop = len(shops[i]) if i in shops else 0
    poi_table.loc[i, 'pub':'shop'] = [pub, bar, restaurant, museum, cinema, art_center, shop]
# poi_table.to_csv('./Gourmet and leisure/Pois.csv')
# %%
# TODO: google API high-end restaurants
# TODO: try Blender and heatmap, area boundary map
