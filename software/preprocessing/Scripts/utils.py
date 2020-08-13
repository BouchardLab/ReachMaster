"""
Functions for data loading, parsing, if you can't find it in the main code its here!
Author: Brett Nelson
"""

import json


def load_exp_dataframe(json_path):
    with open(json_path, encoding='utf-8', errors='ignore') as json_data:
        data = json.load(json_data, strict=False)
    return data

