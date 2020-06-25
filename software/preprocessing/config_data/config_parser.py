import json
import os

def get_config(config_path):
    """

    Parameters
    ----------
    config_path : str
        path to experimental metadata file

    Returns
    -------
    config_file : dict
        dict of experimental metadata from each experiment session
    """
    os.chdir(config_path)
    config_file = str(os.listdir()[0])
    config_file = json.load(open(config_file))
    return config_file


def import_config_data(config_path):
    """

    Parameters
    ----------
    config_path : str
        path to the experimental configuration file

    Returns
    -------
    config data : dict
        dict containing experimental metadata for a given session config file
    """
    data = get_config(config_path)
    return data
