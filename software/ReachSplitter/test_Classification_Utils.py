from unittest import TestCase
import Classification_Utils as CU
import pandas as pd


# how to make unit tests: https://www.jetbrains.com/help/pycharm/testing-your-first-python-application.html#write-test


class TestFileLoading(TestCase):
    def __init__(self):
        """ Initialize unchanged class attributes for testing

        Notes:
            May take a long time to run.
        """
        # load data
        kinematic_data_path = 'tkd16.pkl'
        kinematic_df = CU.pkl_to_df(kinematic_data_path)
        unpickled_list = pd.read_pickle(kinematic_data_path)

    def test_pkl_to_df_NumRows(self):
        """
        Tests:
            each element in 'unpickled_list' corresponds to one row in 'kinematic_df',
        """
        # number of rows in 'kinematic_df' should be same as number of dataframes in 'pickled_list'
        self.assertEqual(len(self.unpickled_list), len(self.kinematic_df))

    def test_pkl_to_df_Index(self):
        """
        Tests:
            returned df is indexed by rat,date,session,dim
        """
        # indexed by rat,date,session,dim
        expected_df_index = ['rat', 'date', 'session', 'dim']
        self.assertEqual(expected_df_index, self.kinematic_df.index.names)

class TestGetMethods(TestCase):
    def __init__(self):
        """ Initialize unchanged class attributes for testing
        """
        # load data
        kinematic_data_path = 'tkd16.pkl'
        kinematic_df = CU.pkl_to_df(kinematic_data_path)

    def test_get_kinematic_trial(self):
        """
        Tests:
            function returned desired trial
        """
        # assumes rat,date,session exist
        rat = 'RM16'
        kdate = '0190920'
        session = 'S3'
        trial_df = CU.get_kinematic_trial(self.kinematic_df, rat, kdate, session)

        # index should correspond to desired trial
        expected_df_index = ('RM16', '0190920', 'S3', 0)
        self.assertEqual(expected_df_index, trial_df.index.values[0])
