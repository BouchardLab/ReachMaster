"""    Written Emily Nguyen, UC Berkeley, NSDS Lab

Function intended to ensure intended functionality of ReachSplitter

reference: how to make unit tests: https://www.jetbrains.com/help/pycharm/testing-your-first-python-application.html#write-test

Edited 6/4/2021 """

from unittest import TestCase

import sklearn
from sklearn.ensemble import RandomForestClassifier
import Classification_Utils as CU
import Trial_Classifier as TC
import pandas as pd
import numpy as np
import os
import joblib  # for saving sklearn models
from imblearn.over_sampling import SMOTE  # for adjusting class imbalances
# classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, RandomizedSearchCV, train_test_split, GridSearchCV, cross_validate
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

current_directory = os.getcwd()


class TestPreprocessing(TestCase):
    """ Test Overview
        - File loading
        - Get and Save Blocks
        - Save all blocks with labels
        - Save all labels
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
        os.remove(final_directory)
        self.assertFalse(os.path.exists(final_directory))

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
        os.remove(final_directory)
        self.assertFalse(os.path.exists(final_directory))

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

    def test_save_all_labeled_blocks(self):
        tkdf_14 = self.preprocessor.load_data('3D_positions_RM14_f.pkl')
        tkdf_15 = self.preprocessor.load_data('tkdf16_f.pkl')

        # choose which rats to save
        save_16 = True
        save_15 = True
        save_14 = True

        if save_16:
            # RM16_9_17_s1
            self.preprocessor.get_single_block(self.exp_data, '0190917', 'S1', 'RM16', format='exp',
                                               save_as=f'{TC.folder_name}/exp_rm16_9_17_s1.pkl')
            self.preprocessor.get_single_block(self.kin_data, '0190917', 'S1', '09172019', format='kin',
                                               save_as=f'{TC.folder_name}/kin_rm16_9_17_s1.pkl')
            # RM16, 9-18, S1
            self.preprocessor.get_single_block(self.exp_data, '0190918', 'S1', 'RM16', format='exp',
                                               save_as=f'{TC.folder_name}/exp_rm16_9_18_s1.pkl')
            self.preprocessor.get_single_block(self.kin_data, '0190918', 'S1', '09182019', format='kin',
                                               save_as=f'{TC.folder_name}/kin_rm16_9_18_s1.pkl')

            # RM16, 9-17, S2
            self.preprocessor.get_single_block(self.exp_data, '0190917', 'S2', 'RM16', format='exp',
                                               save_as=f'{TC.folder_name}/exp_rm16_9_17_s2.pkl')
            self.preprocessor.get_single_block(self.kin_data, '0190917', 'S2', '09172019', format='kin',
                                               save_as=f'{TC.folder_name}/kin_rm16_9_17_s2.pkl')

            # RM16, DATE 9-20, S3
            self.preprocessor.get_single_block(self.exp_data, '0190920', 'S3', 'RM16', format='exp',
                                               save_as=f'{TC.folder_name}/exp_rm16_9_20_s3.pkl')
            self.preprocessor.get_single_block(self.kin_data, '0190920', 'S3', '09202019', format='kin',
                                               save_as=f'{TC.folder_name}/kin_rm16_9_20_s3.pkl')

            # RM16, 09-19-2019, S3
            self.preprocessor.get_single_block(self.exp_data, '0190919', 'S3', 'RM16', format='exp',
                                               save_as=f'{TC.folder_name}/exp_rm16_9_19_s3.pkl')
            self.preprocessor.get_single_block(self.kin_data, '0190919', 'S3', '09192019', format='kin',
                                               save_as=f'{TC.folder_name}/kin_rm16_9_19_s3.pkl')

        if save_15:
            # RM15, 25, S3
            self.preprocessor.get_single_block(self.exp_data, '0190925', 'S3', 'RM15', format='exp',
                                               save_as=f'{TC.folder_name}/exp_rm15_9_25_s3.pkl')
            self.preprocessor.get_single_block(tkdf_15, '0190925', 'S3', '09252019', format='kin',
                                               save_as=f'{TC.folder_name}/kin_rm15_9_25_s3.pkl')

            # RM15, 17, S4
            self.preprocessor.get_single_block(self.exp_data, '0190917', 'S4', 'RM15', format='exp',
                                               save_as=f'{TC.folder_name}/exp_rm15_9_17_s4.pkl')
            self.preprocessor.get_single_block(tkdf_15, '0190917', 'S4', '09172019', format='kin',
                                               save_as=f'{TC.folder_name}/kin_rm15_9_17_s4.pkl')
        if save_14:
            # 2019-09-20-S1-RM14_cam2
            self.preprocessor.get_single_block(self.exp_data, '0190920', 'S1', 'RM14', format='exp',
                                               save_as=f'{TC.folder_name}/exp_rm14_9_20_s1.pkl')
            self.preprocessor.get_single_block(tkdf_14, '0190920', 'S1', '09202019', format='kin',
                                               save_as=f'{TC.folder_name}/kin_rm14_9_20_s1.pkl')

            # 2019-09-18-S2-RM14-cam2
            self.preprocessor.get_single_block(self.exp_data, '0190918', 'S2', 'RM14', format='exp',
                                               save_as=f'{TC.folder_name}/exp_rm14_9_18_s2.pkl')
            self.preprocessor.get_single_block(tkdf_14, '0190918', 'S2', '09182019', format='kin',
                                               save_as=f'{TC.folder_name}/kin_rm14_9_18_s2.pkl')


class TestPreprocessingBlock(TestCase):
    """ Test Overview
        - kin_block median filter
        - test make kin feat df
        - test make exp feat df
    """

    # class variables
    et = 0
    el = 0
    wv = 5
    window_length = 4
    pre = 2

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        """ Initialize instance attributes for testing 
        """
        # load data and update preprocessor object
        self.kin_filename = f'{TC.folder_name}/kin_rm15_9_17_s4.pkl'
        self.exp_filename = f'{TC.folder_name}/exp_rm15_9_17_s4.pkl'
        self.label = CU.rm15_9_17_s4_label
        self.vec_label, _ = CU.make_vectorized_labels(self.label)

        self.preprocessor = TC.Preprocessor()
        self.kin_block = TC.Preprocessor.load_data(self.kin_filename, file_type='pkl')
        self.exp_block = TC.Preprocessor.load_data(self.exp_filename, file_type='pkl')

        # setup preprocessor object
        self.preprocessor.set_exp_block(self.exp_block)
        self.preprocessor.set_wv(self.wv)  # must be set before kin block
        self.preprocessor.set_window_length(self.window_length)
        self.preprocessor.set_pre(self.pre)
        self.preprocessor.set_kin_block(self.kin_block)

        # other
        self.start_frames = self.exp_block['r_start'].values[0]

    def test_median_filter(self):
        # test filtered_df is same size as original
        filtered_df = self.preprocessor.apply_median_filter(self.kin_block, wv=self.wv)
        self.assertEqual(self.kin_block.shape, filtered_df.shape)
        self.assertEqual(self.kin_block.size, filtered_df.size)

    def test_split_kin_trial_0(self):
        # test trial splitting
        trials = TC.Preprocessor.split_trial(self.kin_block, self.exp_block, self.window_length, self.pre)
        trial_0 = trials[0]
        self.assertEqual(len(self.start_frames), len(trials)), "Unequal number of trials!"
        self.assertTrue(trial_0.shape[0] != 0), "Empty trial!"
        self.assertTrue(isinstance(trial_0, pd.DataFrame)), "Not a DF!"

    def test_trialize_kin_blocks_1(self):
        # test reshaping of trials
        trials = TC.Preprocessor.split_trial(self.kin_block, self.exp_block, self.window_length, self.pre)

        ftrials = TC.Preprocessor.trialize_kin_blocks(trials)
        trial_0 = ftrials[0]
        self.assertEqual(len(self.start_frames), len(ftrials)), "Unequal number of trials!"
        self.assertTrue(isinstance(trial_0, pd.DataFrame)), "Not a DF!"
        self.assertTrue(trial_0.shape[0] != 0), "Empty trial!"
        [self.assertTrue(trial.shape[1] == 1) for trial in ftrials]  # check is a one-col df for each trial
        [self.assertFalse(trial.isnull().values.any()) for trial in ftrials]  # check no null values in dfs

    def test_match_kin_2(self):
        # test matching to labels
        trials = TC.Preprocessor.split_trial(self.kin_block, self.exp_block, self.window_length, self.pre)
        ftrials = TC.Preprocessor.trialize_kin_blocks(trials)

        labeled_trials = TC.Preprocessor.match_kin_to_label(ftrials, self.vec_label)
        trial_0 = labeled_trials[0]
        self.assertEqual(len(self.vec_label), len(labeled_trials)), "Not Matching Labels!"
        self.assertTrue(isinstance(trial_0, pd.DataFrame)), "Not a DF!"
        self.assertTrue(trial_0.shape[0] != 0), "Empty trial!"
        [self.assertTrue(trial.shape[0] == 1) for trial in labeled_trials]  # check is a one-row df for each trial
        [self.assertFalse(trial.isnull().values.any()) for trial in ftrials]  # check no null values in dfs

    def test_make_exp(self):
        # tests making exp feat df
        # todo expand?
        exp_features = CU.import_experiment_features(self.exp_block, self.start_frames, self.window_length, self.pre)
        exp_df = CU.import_experiment_features_to_df(exp_features)
        exp_feat_df = TC.Preprocessor.match_exp_to_label(exp_df, self.vec_label)

        self.assertTrue(isinstance(exp_feat_df, pd.DataFrame)), "Not a DF!"
        self.assertEqual(len(self.vec_label), len(exp_feat_df)), "Not Matching Labels!"

    def test_create_kin_feat_df_3(self):
        trials = TC.Preprocessor.split_trial(self.kin_block, self.exp_block, self.window_length, self.pre)
        ftrials = TC.Preprocessor.trialize_kin_blocks(trials)
        labeled_trials = TC.Preprocessor.match_kin_to_label(ftrials, self.vec_label)

        df = TC.Preprocessor.create_kin_feat_df(labeled_trials)
        self.assertEqual(len(self.vec_label), len(df)), "Unequal number of trials!"
        self.assertTrue(isinstance(df, pd.DataFrame)), "Not a DF!"
        self.assertTrue(df.shape[0] != 0), "Empty trial!"


class TestClassificationWorkflow(TestCase):
    """ Test Overview
        - test basic model
    """

    # class variables
    et = 0
    el = 0
    wv = 5
    window_length = 4
    pre = 2

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        """ Initialize unchanged class attributes for testing 
        """
        # load data and update preprocessor object
        self.kin_filename = f'{TC.folder_name}/kin_rm15_9_17_s4.pkl'
        self.exp_filename = f'{TC.folder_name}/exp_rm15_9_17_s4.pkl'
        self.label = CU.rm15_9_17_s4_label
        self.vec_label, _ = CU.make_vectorized_labels(self.label)

        self.preprocessor = TC.Preprocessor()
        self.kin_block = TC.Preprocessor.load_data(self.kin_filename, file_type='pkl')
        self.exp_block = TC.Preprocessor.load_data(self.exp_filename, file_type='pkl')

        # make feat dfs
        self.label_df = CU.make_vectorized_labels_to_df(self.vec_label)
        self.kin_feat_df, self.exp_feat_df = self.preprocessor.make_ml_feat_labels(self.kin_block, self.exp_block,
                                                                                   self.label, self.et, self.el,
                                                                                   self.window_length, self.pre,
                                                                                   self.wv)
        self.X = self.kin_feat_df[self.kin_feat_df.columns[1:4]]
        self.y = self.label_df['Num Reaches'].to_frame()

    def test_basic_make_preds(self):
        # test default model
        model = TC.ReachClassifier(model=RandomForestClassifier())
        model.fit(self.X, self.y)
        self.assertTrue(len(model.predict(self.X)) != 0)  # TODO won't work if label has all of the same class
        # todo and need to fix exp data?

    def test_hyperparam_tuning(self):
        classifier = TC.ReachClassifier()

        # Create first pipeline for base without reducing features.
        model = Pipeline(steps=[('standardscaler', StandardScaler()),
                                ('classifier', RandomForestClassifier())])

        # Create param grid.
        param_grid = [
            {'classifier': [LogisticRegression()],
             'classifier__penalty': ['l1', 'l2'],
             'classifier__C': np.logspace(-4, 4, 20),
             'classifier__solver': ['liblinear']},
            {'classifier': [RandomForestClassifier()],
             'classifier__n_estimators': list(range(10, 101, 10)),
             'classifier__max_features': list(range(6, 32, 5))},
            {'classifier': [sklearn.svm.SVC()],
             'classifier__C': list(range(1, 10, 1))}
        ]

        classifier.hyperparameter_tuning(model, param_grid, self.X, self.y, fullGridSearch=False)



