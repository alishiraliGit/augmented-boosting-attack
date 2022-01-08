import os
import pickle
import csv
from datetime import datetime


def load_knnrecsys_synthetic_data(loadpath, **kwargs):
    with open(os.path.join(loadpath, 'knnrecsys%s.pkl' % stringify(kwargs)), 'rb') as f:
        data_dic = pickle.load(f)

        return data_dic


def get_edge_list_from_file_ml100k(file_path, file_name, do_sort=False):
    # Returned user (item) is actual user (item) - 1
    edges = []
    with open(os.path.join(file_path, file_name)) as data_file:
        data_reader = csv.reader(data_file, delimiter='\t')

        for row in data_reader:
            user, item, value, timestamp = int(row[0]), int(row[1]), float(row[2]), row[3]

            edges += [(user - 1, item - 1, value, datetime.fromtimestamp(int(timestamp)))]
    if do_sort:
        edges = sort_edge_list_by_date(edges)
    return edges


def get_users_attributes_from_file_ml100k(file_path):
    user_dic = {}
    with open(os.path.join(file_path, 'u.user')) as data_file:
        data_reader = csv.reader(data_file, delimiter='|')

        for row in data_reader:
            user, age, sex, occupation = int(row[0]), int(row[1]), row[2], row[3]
            user_dic[user - 1] = {
                'age': age,
                'sex': sex,
                'occupation': occupation
            }

    return user_dic


def get_items_attributes_from_file_ml100k(file_path):
    item_dic = {}
    with open(os.path.join(file_path, 'u.item'), encoding='ISO-8859-1') as data_file:
        data_reader = csv.reader(data_file, delimiter='|')

        for row in data_reader:
            item = int(row[0])
            item_dic[item - 1] = {
                'genres': [int(row[idx]) for idx in range(5, 24)]
            }

    return item_dic


def sort_edge_list_by_date(edges):
    return sorted(edges, key=lambda x: x[3])


def stringify(dic):
    """
    Sorts the input dic by key and returns a stringified version
    :param dic: input dic
    :return: string
    """
    stringified = ''
    for key, val in sorted(dic.items()):
        stringified += '-'
        stringified += key
        if isinstance(val, int):
            stringified += str(val)
        elif isinstance(val, float):
            stringified += '%.0e' % val
        elif isinstance(val, str):
            stringified += val

    return stringified
