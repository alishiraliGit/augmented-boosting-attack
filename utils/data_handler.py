import os
import pickle


def load_knnrecsys_synthetic_data(loadpath, **kwargs):
    with open(os.path.join(loadpath, 'knnrecsys%s.pkl' % stringify(kwargs)), 'rb') as f:
        data_dic = pickle.load(f)

        return data_dic


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
        else:
            stringified += Logger.stringify(val)

    return stringified
