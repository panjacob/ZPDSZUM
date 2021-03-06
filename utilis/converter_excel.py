import json
import os
from pprint import pprint
import csv

types = ['hatchback', 'kombi', 'coupe', 'sedan', 'suv', 'minibus', 'mpv', 'others', 'n/a']
brands = ['toyota', 'volkswagen', 'kia', 'seat', 'hyundai', 'peugeot', 'bmw', 'audi', 'skoda', 'honda', 'renault',
          'opel', 'volvo', 'mazda', 'nissan', 'mercedes', 'citroen', 'ford', 'fiat', 'others', 'n/a']
classes = ['front', 'side', 'rear', 'others']
colors = ['white', 'blackblue', 'silver', 'red', 'gray', 'orange', 'yellow', 'beige', 'others']


def read_csv(filename):
    with open(filename) as f:
        file_data = csv.reader(f)
        headers = next(file_data)[0].lower().split(';')
        result = []
        for x in file_data:
            a = x[0].lower().split(';')
            b = dict(zip(headers, a))
            result.append(b)
        return result


def clean_data(data):
    result = []
    for x in data:
        if 'type' not in x:
            x['type'] = 'n/a'
        if 'brand' not in x:
            x['brand'] = 'n/a'
        if 'color' not in x:
            x['color'] = 'others'
        if 'class' not in x:
            x['class'] = 'others'

        if x['type'] not in types:
            x['type'] = 'n/a'
        if x['brand'] not in brands:
            x['brand'] = 'n/a'
        if x['class'] not in classes:
            x['class'] = 'others'
        if x['color'] not in colors:
            x['color'] = 'others'
        x['images'] = []
        result.append(x)
    return result


def save_json(data, path):
    with open(path, 'w') as fp:
        json.dump(data, fp)


data_raw = read_csv(os.path.dirname(__file__) + os.path.join('/..', 'files', 'classes.csv'))
data_clean = clean_data(data_raw)
pprint(data_clean)
save_json(data_clean, os.path.dirname(__file__) +  os.path.join('/..', 'files', 'classes.json'))
