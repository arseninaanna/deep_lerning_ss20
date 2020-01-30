import pandas as pd
import re
import numpy as np
import random
from math import sin, cos, sqrt, atan2, radians
import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def get_distance(lat1, lon1, lat2, lon2):
    R = 6373.0

    lat1 = radians(lat1)
    lon1 = radians(lon1)
    lat2 = radians(lat2)
    lon2 = radians(lon2)

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    return R * c


def score(path, dict, distances):
    res = 0
    for i in range(len(path) - 1):
        idx1 = dict[path[i]]
        idx2 = dict[path[i + 1]]
        res += distances[idx1][idx2]

    return res


def update_path(path, times=2):
    for i in range(times):
        selected = random.sample(path, k=2)
        city1 = selected[0]
        city2 = selected[1]
        pos1 = path.index(city1)
        pos2 = path.index(city2)
        path[pos1], path[pos2] = path[pos2], path[pos1]

    return path


def acceptance_ratio(cost, new_cost, temperature):
    p = np.exp(- cost / temperature)
    p_new = np.exp(- new_cost / temperature)

    if p == 0:
         return 0

    return p_new / p


def simpleSA(T, rate, path, distances, idx_dict):
    cost = score(path, idx_dict, distances)

    cost_record = 0
    iter = 0
    while True:
        if iter % 5000 == 0:
            if cost_record == cost:
                break

            cost_record = cost
            print(cost_record)

        old_path = path.copy()
        path = update_path(path)
        new_cost = score(path, idx_dict, distances)

        alpha = acceptance_ratio(cost, new_cost, T)
        u = np.random.uniform(0, 1, 1)[0]
        if u <= alpha:
            cost = new_cost
        else:
            path = old_path

        T *= rate
        iter += 1

topN = 30

data = pd.read_csv("cities.csv")
data['population'] = data['population'].transform(lambda x: int(re.sub('\[\d\]', '', str(x))))
data['geo_lat'] = data['geo_lat'].astype('float')
data['geo_lon'] = data['geo_lon'].astype('float')

data.sort_values("population", axis=0, ascending=False,
                 inplace=True, na_position='last')

data.city.fillna(data.region, inplace=True)
df = data[['city', 'population', 'geo_lat', 'geo_lon']]
df = df[:topN]

cities = list(df['city'])
cities_to_idx = {}
idx_to_cities = {}
for i in range(len(cities)):
    cities_to_idx[cities[i]] = i
    idx_to_cities[i] = cities[i]

cities_dist = np.zeros((topN, topN))
for i in range(topN):
    for j in range(i):
        if i != j:
            city_a = df.loc[df['city'] == idx_to_cities[i]]
            city_b = df.loc[df['city'] == idx_to_cities[j]]

            distance = get_distance(float(city_a['geo_lat']), float(city_a['geo_lon']), float(city_b['geo_lat']),
                                    float(city_b['geo_lon']))

            cities_dist[i][j] = distance
            cities_dist[j][i] = distance

random.shuffle(cities)
start_time = time.time()
simpleSA(10000000000, 0.99, cities, cities_dist, cities_to_idx)
print("--- %s seconds ---" % (time.time() - start_time))
