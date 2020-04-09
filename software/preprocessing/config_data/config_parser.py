


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
    files = [i for i in os.listdir(config_path) if os.path.isfile(os.path.join(config_path, i)) and \
             'Workspace%' in i]
    os.chdir(config_path)
    config_file = json.load(open(files[0]))
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
        dict containing relevant experimental metadata
    """
    data = get_config(config_path)
    config_data = {'command__file': data['RobotSettings']['commandFile'], 'x_pos': data['RobotSettings']['xCommandPos'],
                   'y_pos': data['RobotSettings']['yCommandPos'], 'z_pos': data['RobotSettings']['zCommandPos']}
    return config_data