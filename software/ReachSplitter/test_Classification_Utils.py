"""    Written Emily Nguyen, UC Berkeley, NSDS Lab

Function intended to ensure intended functionality of ReachSplitter

reference: how to make unit tests: https://www.jetbrains.com/help/pycharm/testing-your-first-python-application.html#write-test

Edited 6/4/2021 """

from unittest import TestCase
from sklearn.ensemble import RandomForestClassifier
import Classification_Utils as CU
import Trial_Classifier as TC
import pandas as pd
import numpy as np
import os

current_directory = os.getcwd()


class TestPreprocessing(TestCase):
    """ Test Overview
        - File loading
        - Get and Save Block
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        """ Initialize instance attributes for testing
        """
        # load data and update preprocessor object
        self.exp_filename = 'experimental_data.pickle'
        self.kin_filename = 'tkdf16_f.pkl'
        self.preprocessor = TC.Preprocessor()

        self.kin_data = self.preprocessor.load_data(self.kin_filename)
        self.preprocessor.set_kin_data(self.kin_data)

        self.exp_data = self.preprocessor.load_data(self.exp_filename)
        self.preprocessor.set_exp_data(self.exp_data)

    def test_basic_load_blocks(self):
        # test dataframes are not empty
        self.assertTrue(len(self.exp_data.index) != 0)
        self.assertTrue(len(self.kin_data) != 0)

    def test_load_assertion(self):
        # tests assertion is raised with invalid file_type
        try:
            TC.Preprocessor.load_data("some file name", file_type='ABC')
        except AssertionError:
            print('pass: load_data assertion')
            pass

    def test_get_exp_block(self):
        # tests get exp block
        date = '0190918'
        session = 'S1'
        rat = 'RM16'
        filename = f'{TC.folder_name}/exp_{rat}{date}{session}.pkl'
        self.preprocessor.get_single_block(self.exp_data, date, session, rat, format='exp', save_as=filename)

        # tests save exp block
        final_directory = os.path.join(current_directory, filename)
        self.assertTrue(os.path.exists(final_directory))

    def test_get_kin_block(self):
        # test get kin block
        rat = '09182019'  # RM16
        date = '0190918'
        session = 'S1'
        filename = f'{TC.folder_name}/kin_{rat}{date}{session}.pkl'
        df = self.preprocessor.get_single_block(self.kin_data, date, session, rat, format='kin', save_as=filename)
        matching_index = df[rat][session][date][0]
        self.assertTrue(len(df.index) != 0)

        # test save kin block
        final_directory = os.path.join(current_directory, filename)
        self.assertTrue(os.path.exists(final_directory))

    def test_get_kin_block2(self):
        date = '0190920'
        session = 'S4'
        rat = '09202019'
        df = self.preprocessor.get_single_block(self.preprocessor.kin_data, date, session, rat, format='kin')
        matching_index = df[rat][session][date][0]
        self.assertTrue(len(df.index) != 0)

    def test_get_assertion(self):
        # tests assertion is raised with invalid block
        try:
            self.preprocessor.get_single_block(self.kin_data, '00000', 'S3', '111111', format='kin')
        except AssertionError:
            print('pass: get_data assertion')
            pass



    def test_pkl_to_df_NumRows(self):
        """
        Tests:
            each element in 'unpickled_list' corresponds to one row in 'kinematic_df',
        """
        # number of rows in 'kinematic_df' should be same as number of dataframes in 'pickled_list'
        # self.assertEqual(len(self.unpickled_list), len(self.kinematic_df))
        pass

    def test_pkl_to_df_Index(self):
        """
        Tests:
            returned df is indexed by rat,date,session,dim
        """
        # indexed by rat,date,session,dim
        # expected_df_index = ['rat', 'date', 'session', 'dim']
        # self.assertEqual(expected_df_index, self.kin_data.index.names)
        pass


class TestPreprocessingBlock(TestCase):
    """ Test Overview
        - kin_block median filter
        -
    """

    # class variables
    et = 0
    el = 0
    wv = 5
    window_length = 4  # TODO made small,run final on default = 250
    pre = 2  # TODO  made small,run final on default = 10

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        """ Initialize instance attributes for testing 
        """
        # load data and update preprocessor object
        self.kin_filename = f'{TC.folder_name}/kin_091820190190918S1.pkl'
        self.exp_filename = f'{TC.folder_name}/exp_RM160190918S1.pkl'
        self.label = CU.rm16_9_18_s1_label

        self.preprocessor = TC.Preprocessor()
        self.kin_block = self.preprocessor.load_data(self.kin_filename, file_type='pkl')
        self.exp_block = self.preprocessor.load_data(self.exp_filename, file_type='pkl')

    def test_median_filter(self):
        # test filtered_df is same size as original
        filtered_df = self.preprocessor.apply_median_filter(self.kin_block, wv=self.wv)
        self.assertEqual(self.kin_block.shape, filtered_df.shape)
        self.assertEqual(self.kin_block.size, filtered_df.size)

    def test_trialize_1(self):
        feat_df = self.preprocessor.make_ml_feat_labels(self.kin_block, self.exp_block, self.label,
                                                        self.et, self.el, self.window_length, self.pre, self.wv)
        model = TC.IsReachClassifier()
        label_df = CU.make_vectorized_labels_to_df(self.preprocessor.label)
        X = feat_df[feat_df.columns[1:4]]
        y = label_df['Num Reaches'].to_frame()
        model.fit(X, y)
        self.assertTrue(len(model.predict(X)) != 0)   #TODO won't work if label has all of the same class and need to fix exp data!

class TestClassificationWorkflow(TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        """ Initialize unchanged class attributes for testing 
        """
        # load data
        kinematic_data_path = 'tkd16.pkl'
        robot_data_path = 'experimental_data.pickle'
        self.kinematic_df = CU.pkl_to_df(kinematic_data_path)
        self.robot_df = pd.read_pickle(robot_data_path)

    def test_make_s_f_trial_arrays_from_block(self):
        """
        Tests: Function and its helpers return non-empty values
        """
        # Vectorize DLC labels into ML ready format
        elists, ev = CU.make_vectorized_labels(elist)
        labellist, edddd = CU.make_vectorized_labels(blist1)
        nl1lists, ev1 = CU.make_vectorized_labels(nl1)
        nl2lists, ev2 = CU.make_vectorized_labels(nl2)
        l18l, ev18 = CU.make_vectorized_labels(l18)

        # Define variables
        et = 0
        el = 0

        # Trialize data
        d = self.kinematic_df
        hdf = self.robot_df
        hot_vector, tt, feats, e = CU.make_s_f_trial_arrays_from_block(d, hdf, et, el, 'RM16', '0190920', '0190920',
                                                                       'S3', 9, window_length=800, pre=100)
        hot_vector3, tt3, feats3, e3 = CU.make_s_f_trial_arrays_from_block(d, hdf, et, el, 'RM16', '0190919', '0190919',
                                                                           'S3', 9)  # Emily label trial list
        hot_vectornl2, ttnl2, featsnl2, enl2 = CU.make_s_f_trial_arrays_from_block(d, hdf, et, el, 'RM16', '0190917',
                                                                                   '0190917', 'S2', 9)
        hot_vectornl1, ttnl1, featsnl1, enl1 = CU.make_s_f_trial_arrays_from_block(d, hdf, et, el, 'RM16', '0190917',
                                                                                   '0190917', 'S1', 9)
        hot_vectorl18, ttl18, featsl18, el18 = CU.make_s_f_trial_arrays_from_block(d, hdf, et, el, 'RM16', '0190918',
                                                                                   '0190918', 'S1', 9)

        # check return values exist
        self.assertNotEqual(0, len(hot_vectorl18))
        self.assertNotEqual(0, len(ttnl1))
        self.assertNotEqual(0, len(e))
        self.assertNotEqual(0, len(feats3))

    def test_trial_workflow_pipeline(self):
        """
        Tests: Generation of final ML and feature values its helpers return non-empty values

        Notes:
            Code is same as that found in ipynb
        """
        # Vectorize DLC labels into ML ready format
        elists, ev = CU.make_vectorized_labels(elist)
        labellist, edddd = CU.make_vectorized_labels(blist1)
        nl1lists, ev1 = CU.make_vectorized_labels(nl1)
        nl2lists, ev2 = CU.make_vectorized_labels(nl2)
        l18l, ev18 = CU.make_vectorized_labels(l18)

        # Define variables
        et = 0
        el = 0

        # Trialize data
        d = self.kinematic_df
        hdf = self.robot_df
        hot_vector, tt, feats, e = CU.make_s_f_trial_arrays_from_block(d, hdf, et, el, 'RM16', '0190920', '0190920',
                                                                       'S3', 9, window_length=250, pre=10)
        hot_vector3, tt3, feats3, e3 = CU.make_s_f_trial_arrays_from_block(d, hdf, et, el, 'RM16', '0190919', '0190919',
                                                                           'S3', 9)  # Emily label trial list
        hot_vectornl2, ttnl2, featsnl2, enl2 = CU.make_s_f_trial_arrays_from_block(d, hdf, et, el, 'RM16', '0190917',
                                                                                   '0190917', 'S2', 9)
        hot_vectornl1, ttnl1, featsnl1, enl1 = CU.make_s_f_trial_arrays_from_block(d, hdf, et, el, 'RM16', '0190917',
                                                                                   '0190917', 'S1', 9)
        hot_vectorl18, ttl18, featsl18, el18 = CU.make_s_f_trial_arrays_from_block(d, hdf, et, el, 'RM16', '0190918',
                                                                                   '0190918', 'S1', 9)
        # Match
        matched_kin_b1, ez1 = CU.match_stamps(tt, blist1, e)  # Matched to blist1
        matched_kin_e1, ez4 = CU.match_stamps(tt3, elists, e3)  # Matched to blist4
        matched_kin_b1nl1, eznl1 = CU.match_stamps(ttnl1, nl1lists, enl1)  # Matched to blist1
        matched_kin_b1nl2, eznl2 = CU.match_stamps(ttnl2, nl2lists, enl2)  # Matched to blist1
        matched_kin_e1l18, ezl18 = CU.match_stamps(ttl18, l18l, el18)  # Matched to blist4

        # Slice, reshape, and concat arrays
        c = CU.create_ML_array(matched_kin_b1, ez1)
        c1 = CU.create_ML_array(matched_kin_e1, ez4)
        c2 = CU.create_ML_array(matched_kin_e1l18, ezl18)
        c3 = CU.create_ML_array(matched_kin_b1nl1, eznl1)
        c4 = CU.create_ML_array(matched_kin_b1nl2, eznl2)

        # Create final ML arrays
        final_ML_array, final_feature_array = CU.stack_ML_arrays([c, c1, c2, c3, c4],
                                                                 [blist1, elists, l18l, nl1lists, nl2lists])

        # check return values exist
        self.assertNotEqual(0, len(final_feature_array))
        self.assertNotEqual(0, len(final_feature_array))


#####
# DLC labels
# accessible to all testing classes
#####
nl1 = [
    [1, 0, 0, 1, 1, 'l', 'Tug', 0, 0],  # single left rew tug
    [2, 0, 0, 1, 1, 'l', 'Tug', 0, 0],  # single left rew tug
    [3, 0, 0, 1, 1, 'rla', 'noTug', 0, 0],  # rl assist, 1 , notug, rew
    [4, 0, 0, 1, 1, 'l', 'noTug', 0, 0],  # l single rew notug
    [5, 0, 0, 1, 2, 'l', 'noTug', 0, 0],  # l 2 rew no tug
    [6, 0, 0, 1, 2, 'lra', 'Tug', 0, 0],  # lra 2 rew tug
    [7, 0, 0, 1, 4, 'lra', 'noTug', 0, 0],  # lra 4 rew notug
    [8, 0, 0, 1, 1, 'lra', 'noTug', 0, 0],  # lra 1 rew notug
    [9, 0, 0, 1, 1, 'l', 'Tug', 0, 0],  # l 1 rew tug
    [10, 0, 0, 1, 1, 'l', 'Tug', 0, 0],  # l 1 rew tug
    [11, 0, 0, 1, 1, 'lra', 'Tug', 0, 0],  # lra 2 rew tug
    [12, 0, 0, 1, 3, 'lra', 'Tug', 0, 0],  # lra 3 rew tug
    [13, 0, 0, 1, 3, 'lra', 'Tug', 0, 0],  # lra 3 rew tug
    [14, 0, 0, 1, 1, 'bi', 'Tug', 0, 0],  # bi 1 rew tug
    [15, 0, 0, 1, 1, 'l', 'Tug', 0, 0],  # l 1 rew tug
    [16, 0, 0, 1, 1, 'l', 'Tug', 0, 0],  # l 1 rew tug
    [17, 0, 0, 1, 1, 'lra', 'Tug', 0, 0],  # lra 1 rew tug
    [19, 0, 0, 1, 2, 'bi', 'Tug', 0, 0],  # bi 2 rew tug
    [20, 0, 0, 1, 1, 'l', 'Tug', 0, 0],  # l 1 r t
    [21, 0, 0, 1, 1, 'bi', 'Tug', 0, 0],  # bi 1 r t
    [22, 0, 0, 1, 1, 'l', 'Tug', 0, 0],  # l 1 r t
    [23, 0, 0, 1, 1, 'lra', 'Tug', 0, 0],  # lra 1 r t
    [24, 0, 0, 1, 1, 'l', 'Tug', 0, 0],  # l 1 r t
    [25, 0, 0, 1, 1, 'lra', 'Tug', 0, 0],  # lr 1 r t
    [26, 0, 0, 1, 1, 'l', 'Tug', 0, 0],  # l 1 r t
    [18, 0, 0, 1, 1, 'lra', 'Tug', 0, 0],  # lr 1 r t
    [0, 0, 0, 1, 2, 'rla', 'Tug', 1, 0]  # rl 2 r t hand switch

]

l18 = [
    [1, 0, 0, 1, 1, 'l', 'noTug', 0, 0],  # l 1 rew notug
    [2, 0, 0, 1, 3, 'l', 'noTug', 0, 0],  # l 3 rew notug
    [4, 0, 0, 1, 3, 'lr', 'noTug', 1, 0],  # lr 3 switching rew notug
    [5, 0, 0, 1, 1, 'l', 'noTug', 0, 0],  # l 1 rew no tug
    [6, 0, 0, 1, 1, 'l', 'noTug', 0, 0],  # l 1 rew no tug
    [7, 0, 0, 1, 2, 'lra', 'noTug', 0, 0],  # lr 2 rew notug
    [9, 0, 0, 1, 1, 'l', 'noTug', 0, 0],  # l 1 rew notug (check b4)
    [10, 0, 0, 1, 1, 'lra', 'noTug', 0, 0],  # lr 1 rew notug
    [12, 0, 0, 1, 1, 'lr', 'noTug', 0, 0],  # lr 1 rew notug
    [14, 0, 0, 1, 1, 'lr', 'Tug', 0, 0],  # lr 1 rew tug
    [15, 0, 0, 1, 1, 'l', 'noTug', 0, 0],  # l 1 rew notug
    [17, 0, 0, 1, 1, 'lra', 'noTug', 0, 0],  # lr 1 rew notug
    [0, 0, 0, 1, 4, 'l', 'noTug', 0, 0],  # l 4 norew notug
    [3, 0, 0, 1, 4, 'lra', 'noTug', 1, 0],  # lr 4 switching norew notug
    [8, 0, 0, 1, 11, 'lra', 'noTug', 1, 0],  # lr 11 switching norew notug
    [11, 0, 0, 1, 7, 'l', 'noTug', 0, 0],  # l 7 norew notug
    [13, 0, 0, 1, 7, 'l', 'noTug', 0, 0],  # l 7 norew notug
    [16, 0, 0, 1, 2, 'l', 'noTug', 0, 0],  # l 2 norew notug
    [18, 0, 0, 1, 6, 'l', 'noTug', 0, 0]
]

nl2 = [
    [1, 0, 0, 1, 1, 'l', 'noTug', 0, 0],  # l 1 rew notug
    [2, 0, 0, 1, 1, 'lra', 'Tug', 0, 0],  # lr 1 rew tug
    [3, 0, 0, 1, 2, 'lra', 'Tug', 0, 0],  # lr 2 rew tug
    [4, 0, 0, 1, 1, 'lra', 'Tug', 0, 0],  # lr 1 rew tug
    [5, 0, 0, 1, 1, 'lra', 'Tug', 0, 0],  # lr 1 rew tug
    [6, 0, 0, 1, 2, 'lra', 'Tug', 0, 0],  # lr 2 rew tug
    [7, 0, 0, 1, 1, 'lra', 'Tug', 0, 0],  # lr 1 rew tug
    [8, 0, 0, 1, 2, 'l', 'Tug', 0, 0],  # l 2 rew tug
    [9, 0, 0, 1, 3, 'l', 'Tug', 0, 0],  # l 3 rew tug
    [10, 0, 0, 1, 2, 'l', 'noTug', 0, 0],  # l 2 rew nt
    [11, 0, 0, 1, 2, 'l', 'noTug', 0, 0],  # l 2 rew nt
    [12, 0, 0, 1, 2, 'lra', 'noTug', 0, 0],  # lr 2 rew nt
    [13, 0, 0, 1, 2, 'lra', 'noTug', 0, 0],  # lr 2 rew nt
    [14, 0, 0, 1, 1, 'l', 'Tug', 0, 0],  # l 1 rew t
    [15, 0, 0, 1, 1, 'l', 'Tug', 0, 0],  # l 1 rew t
    [16, 0, 0, 1, 1, 'lra', 'noTug', 0, 0],  # lr 1 rew nt
    [17, 0, 0, 1, 1, 'l', 'noTug', 0, 0],  # l 1 rew nt
    [18, 0, 0, 1, 2, 'l', 'noTug', 0, 0],  # l 2 rew nt
    [19, 0, 0, 1, 1, 'l', 'Tug', 0, 0],  # l 1 rew t
    [20, 0, 0, 1, 2, 'l', 'noTug', 0, 0],  # l 2 rew nt
    [21, 0, 0, 1, 1, 'lra', 'Tug', 0, 0],  # lr 1 rew t
    [22, 0, 0, 1, 1, 'lra', 'Tug', 0, 0],  # lr 1 rew t
    [23, 0, 0, 1, 1, 'l', 'Tug', 0, 0],  # l 1 rew t
    [24, 0, 0, 1, 1, 'l', 'Tug', 0, 0],  # l 1 rew t
    [25, 0, 0, 1, 1, 'l', 'Tug', 0, 0],  # l 1 rew t
    [26, 0, 0, 1, 1, 'l', 'Tug', 0, 0],  # l 1 rew t
    [27, 0, 0, 1, 1, 'l', 'Tug', 0, 0],  # lr 1 rew t
    [28, 0, 0, 1, 6, 'lra', 'noTug', 1, 0],  # lr 6 handswitch rew nt
    [30, 0, 0, 1, 15, 'lra', 'Tug', 0, 0],  # lr 15 rew t
    [31, 0, 0, 1, 1, 'l', 'Tug', 0, 0],  # l 1 rew t
    [32, 0, 0, 1, 1, 'l', 'noTug', 0, 0],  # l 1 rew nt
    [33, 0, 0, 1, 1, 'lra', 'noTug', 0, 0],  # lr 1 rew nt
    [34, 0, 0, 1, 3, 'l', 'Tug', 0, 0],  # l 3 rew t
    [35, 0, 0, 1, 1, 'l', 'Tug', 0, 0],  # l 1 rew t
    [36, 0, 0, 1, 3, 'l', 'Tug', 0, 0],  # l 3 rew t
    [0, 0, 0, 1, 1, 'lra', 'Tug', 0, 0],  # lr 1 rew t
    [29, 0, 0, 1, 3, 'l', 'noTug', 0, 0],  # l 3 nr nt
    [37, 0, 0, 1, 1, 'l', 'Tug', 0, 0],  # l 1 r t
]

blist1 = [
    [1, 1, 1, 4, 1, 'l', 'Tug', 0, 0]  # succ tugg, left hand, single reach,
    , [2, 1, 1, 3, 1, 'l', 'noTug', 0, 0]  # left handed reach, no tug of war, 1 reach, no switch,
    , [3, 1, 1, 3, 2, 'bi', 0, 1, 0]
    , [4, 1, 1, 4, 1, 'l', 'Tug', 0, 0]  # tries to grab handle when moving
    , [6, 1, 1, 3, 1, 'l', 'Tug', 0, 0]  # reaching after handle moves but can't grasp
    , [7, 1, 1, 2, 2, 'l', 0, 0, 0]  # A mis-label!!
    , [8, 1, 1, 3, 1, 'l', 'noTug', 0, 0]
    , [10, 1, 1, 3, 2, 'l', 'Tug', 0, 0]
    , [17, 1, 1, 4, 1, 'l', 'Tug', 0, 0]  #
    , [18, 1, 1, 3, 2, 'lbi', 'noTug', 1, 0]  # added lbi for multiple reaching arms
    , [19, 1, 1, 3, 2, 'lbi', 'noTug', 1, 0]  # lbi
    , [20, 1, 1, 4, 3, 'l', 'Tug', 0, 0]
    , [21, 1, 1, 4, 2, 'lbi', 'Tug', 0, 0]
    , [22, 1, 1, 3, 1, 'l', 'noTug', 0, 0]
    , [23, 1, 1, 3, 1, 'l', 'noTug', 0, 0]
    , [24, 1, 1, 3, 2, 'lbi', 'noTug', 0, 0]
    , [25, 1, 1, 2, 4, 'l', 0, 0, 0]
    , [26, 1, 1, 3, 3, 'lbi', 'noTug', 1, 0]
    , [27, 1, 1, 4, 1, 'lbi', 'Tug', 0, 0]
    , [28, 1, 1, 3, 2, 'l', 'noTug', 0, 0]
    , [30, 1, 1, 3, 2, 'l', 'noTug', 0, 0]
    , [31, 1, 1, 3, 2, 'lbi', 'noTug', 0, 0]
    , [32, 1, 1, 2, 4, 'l', 0, 0, 0]
    , [33, 1, 1, 4, 2, 'lr', 'noTug', 1, 0]
    , [34, 1, 1, 3, 1, 'lbi', 'noTug', 0, 0]
    , [35, 1, 1, 3, 1, 'l', 'noTug', 0, 0]
    , [37, 1, 1, 4, 1, 'lbi', 'Tug', 0, 0]
    , [38, 1, 1, 3, 2, 'lbi', 'noTug', 0, 0]
    , [39, 1, 1, 3, 3, 'l', 'noTug', 0, 0]
    , [40, 1, 1, 3, 1, 'l', 'noTug', 0, 0]
    , [43, 1, 1, 3, 1, 'bi', 'noTug', 0, 0]
    , [44, 1, 1, 4, 2, 'lbi', 'Tug', 0, 0]
    , [45, 1, 1, 3, 1, 'lbi', 'noTug', 0, 0]
    , [46, 1, 1, 3, 1, 'lbi', 'noTug', 0, 0]
    , [47, 1, 1, 3, 1, 'lbi', 'noTug', 0, 0]
    , [48, 1, 1, 3, 2, 'lbi', 'Tug', 0, 0]
    , [0, 1, 1, 0, 0, 0, 0, 0, 0]
    , [5, 1, 1, 2, 4, 'lr', 0, 0, 0]
    , [9, 1, 1, 1, 3, 'l', 0, 0, 0]
    , [11, 1, 1, 1, 1, 'l', 0, 0, 0]
    , [12, 1, 1, 1, 1, 'l', 0, 0, 0]
    , [13, 1, 1, 1, 1, 'l', 0, 0, 0]
    , [14, 1, 1, 1, 1, 'l', 0, 0, 0]
    , [15, 1, 1, 1, 1, 'l', 0, 0, 0]
    , [16, 1, 1, 0, 0, 0, 0, 0, 0]
    , [29, 1, 1, 1, 4, 'l', 0, 0, 0]
    , [36, 1, 1, 1, 9, 'llr', 'Tug', 1, 0]  # lots of stuff going on here
    , [41, 1, 1, 1, 6, 'l', 0, 0, 0]
    , [42, 1, 1, 1, 8, 'llr', 'Tug', 1, 0]
    , [49, 1, 1, 2, 4, 'lr', 0, 0, 0]]

elist = [
    [0, 1723, 2284, 0, 0, 0, 'no_tug', 0, 30]  # null
    , [1, 5593, 6156, 0, 0, 0, 'no_tug', 0, 27]  # null
    , [2, 7866, 8441, 3, 2, 'l', 'no_tug', 0, 14]  # success
    , [3, 8873, 9426, 1, 7, 'l', 'no_tug', 0, 20]  # failed
    , [4, 10101, 10665, 1, 3, 'l', 'no_tug', 0, 15]  # failed
    , [5, 12962, 13524, 1, 8, 'l', 'no_tug', 0, 27]  # failed

    , [6, 14760, 15351, 3, 2, 'bi', 'no_tug', 1, 25]  # success ## bi # starts mid reach
    , [7, 15802, 16431, 3, 3, 'bi', 'no_tug', 1, 30]  # success ## bi # starts mid reach # post reaching activity
    , [8, 17400, 17964, 1, 3, 'l', 'no_tug', 0, 13]  # failed # starts mid reach
    , [9, 18923, 19485, 3, 4, 'l', 'no_tug', 0, 19]  # success
    , [10, 20044, 20604, 1, 5, 'l', 'no_tug', 0, 6]  # failed
    , [11, 24406, 24969, 1, 1, 'l', 'no_tug', 0, 6]  # failed # ends mid reach
    , [12, 26962, 27521, 3, 1, 'l', 'no_tug', 0, 5]  # success # starts mid reach
    , [13, 27980, 28536, 1, 12, 'l', 'no_tug', 0, 18]  # failed # ends mid reach # lots of reaches
    , [14, 29034, 29596, 3, 6, 'bi', 'no_tug', 1, 13]  # success ## bi
    , [15, 30106, 30665, 3, 1, 'l', 'no_tug', 0, 8]  # success # starts mid reach
    , [16, 38998, 39591, 1, 2, 'l', 'no_tug', 0, 4]  # failed
    , [17, 40033, 40594, 0, 0, 0, 'no_tug', 0, 32]  # null
    , [18, 45355, 45914, 3, 7, 'l', 'no_tug', 0, 6]  # success
    , [19, 46845, 47405, 3, 1, 'l', 'no_tug', 0, 7]  # success

    , [20, 50359, 50949, 3, 1, 'l', 'no_tug', 1, 8]  # success # post reaching activity with r
    , [21, 58229, 58793, 3, 2, 'l', 'tug', 1, 12]
    # success # post reaching activity with r # rat lets handle go before in reward zone
    , [22, 59596, 60427, 3, 2, 'l', 'no_tug', 0, 9]  # success
    , [23, 60903, 61466, 3, 1, 'l', 'no_tug', 0, 4]  # success
    , [24, 62233, 62790, 3, 2, 'l', 'tug', 0, 10]  # success # rat lets handle go before in reward zone
    , [25, 66026, 66600, 1, 9, 'l', 'no_tug', 0, 27]
    # classifed as success in py notebook, but is failed trial # ends mid reach
    , [26, 67473, 68046, 3, 1, 'l', 'no_tug', 1, 7]  # success # post reaching activity with r
    , [27, 68689, 69260, 3, 2, 'bi', 'no_tug', 1, 9]  # success # bi
    , [28, 70046, 70617, 3, 2, 'bi', 'no_tug', 1, 5]  # success # bi # starts mid reach

    , [29, 71050, 71622, 3, 11, 'bi', 'tug', 1, 7]
    # success # bi # starts mid reach # rat lets handle go before in reward zone # lots of reaches
    , [30, 72914, 73501, 3, 1, 'l', 'no_tug', 0, 10]  # success
    , [31, 74777, 75368, 3, 3, 'bi', 'no_tug', 1, 9]  # success # bi # post reaching activity with r
    , [32, 81538, 82106, 3, 9, 'l', 'no_tug', 1, 13]  # success # post reaching activity with r
    , [33, 82534, 83114, 3, 4, 'bi', 'tug', 1, 12]
    # success ## bi # starts mid reach # rat lets handle go before in reward zone # includes uncommon failed bi reach
    , [34, 83546, 84118, 3, 2, 'l', 'no_tug', 1, 4]  # success # starts mid reach # post reaching activity with r
    , [35, 85563, 86134, 3, 2, 'l', 'no_tug', 1, 5]  # success # starts mid reach # post reaching activity with r
    , [36, 86564, 87134, 1, 13, 'l', 'no_tug', 0, 5]  # fail # lots of reaches
    , [37, 87574, 88173, 3, 7, 'l', 'no_tug', 1, 8]  # success # post reaching activity with r
    , [38, 89012, 89584, 3, 4, 'bi', 'tug', 1, 5]
    # success ## bi # rat lets handle go before in reward zone # includes uncommon reach with r first then left in bi reach

    , [39, 90738, 91390, 3, 7, 'l', 'no_tug', 1, 9]  # success # post reaching activity with r
    , [40, 91818, 92387, 1, 7, 'l', 'no_tug', 0, 6]  # fail # starts mid reach
]
