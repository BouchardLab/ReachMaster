"""    Written Emily Nguyen, UC Berkeley, NSDS Lab

Function intended to ensure intended functionality of ReachSplitter

reference: how to make unit tests: https://www.jetbrains.com/help/pycharm/testing-your-first-python-application.html#write-test

Edited 6/4/2021 """

from unittest import TestCase

import random
import sklearn
from PycharmProjects.ReachMaster.software.ReachSplitter.Classifier_Training import Classification_Utils as CU
import Trial_Classifier_Training as TC
import pandas as pd
import numpy as np
import os
# classification
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression, RidgeClassifier

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
            tkdf_15 = self.preprocessor.load_data('3D_positions_RM15_f.pkl')
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
            tkdf_14 = self.preprocessor.load_data('3D_positions_RM14_f.pkl')
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
        self.kin_filename = f'{TC.folder_name}/kin_rm16_9_17_s2.pkl'
        self.exp_filename = f'{TC.folder_name}/exp_rm16_9_17_s2.pkl'
        self.label = CU.rm16_9_17_s2_label
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
        trials, times = TC.Preprocessor.split_trial(self.kin_block, self.exp_block, self.window_length, self.pre)
        trial_0 = trials[0]
        self.assertEqual(len(self.start_frames), len(trials)), "Unequal number of trials!"
        self.assertTrue(trial_0.shape[0] != 0), "Empty trial!"
        self.assertTrue(isinstance(trial_0, pd.DataFrame)), "Not a DF!"

    def test_trialize_kin_blocks_1(self):
        # test reshaping of trials
        trials, times = TC.Preprocessor.split_trial(self.kin_block, self.exp_block, self.window_length, self.pre)

        ftrials = TC.Preprocessor.trialize_kin_blocks(trials)
        trial_0 = ftrials[0]
        self.assertEqual(len(self.start_frames), len(ftrials)), "Unequal number of trials!"
        self.assertTrue(isinstance(trial_0, pd.DataFrame)), "Not a DF!"
        self.assertTrue(trial_0.shape[0] != 0), "Empty trial!"
        [self.assertTrue(trial.shape[1] == 1) for trial in ftrials]  # check is a one-col df for each trial
        [self.assertFalse(trial.isnull().values.any()) for trial in ftrials]  # check no null values in dfs

    def test_match_kin_2(self):
        # test matching to labels
        trials,times = TC.Preprocessor.split_trial(self.kin_block, self.exp_block, self.window_length, self.pre)
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
        exp_features = CU.import_experiment_features(self.exp_block, self.start_frames, self.window_length, self.pre)
        exp_df = CU.import_experiment_features_to_df(exp_features)
        exp_feat_df = TC.Preprocessor.match_exp_to_label(exp_df, self.vec_label)

        self.assertTrue(isinstance(exp_feat_df, pd.DataFrame)), "Not a DF!"
        self.assertEqual(len(self.vec_label), len(exp_feat_df)), "Not Matching Labels!"

    def test_create_kin_feat_df_3(self):
        trials,times = TC.Preprocessor.split_trial(self.kin_block, self.exp_block, self.window_length, self.pre)
        ftrials = TC.Preprocessor.trialize_kin_blocks(trials)
        labeled_trials = TC.Preprocessor.match_kin_to_label(ftrials, self.vec_label)

        df = TC.Preprocessor.create_kin_feat_df(labeled_trials)
        self.assertEqual(len(self.vec_label), len(df)), "Unequal number of trials!"
        self.assertTrue(isinstance(df, pd.DataFrame)), "Not a DF!"
        self.assertTrue(df.shape[0] != 0), "Empty trial!"

    def test_preprocess_blocks(self):
        pass


class TestClassificationWorkflow(TestCase):
    """ Test Overview
        - test basic model
        - test hyperparameter tuning
        - test class imbalance
        - test feature selection, plot feature selection
        - test train and validate
        - test make single split predictions
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
        # load data
        self.preprocessor = TC.Preprocessor()
        all_kin_features = self.preprocessor.load_data(f'{TC.folder_name}/kin_feat.pkl', file_type='pkl')  # generate via TC.main
        all_exp_features = self.preprocessor.load_data(f'{TC.folder_name}/exp_feat.pkl')
        all_exp_features = TC.ReachClassifier().mean_df(all_exp_features)
        all_exp_features.dropna(axis=1, inplace=True)
        all_label_dfs = self.preprocessor.load_data(f'{TC.folder_name}/label_dfs.pkl')
        self.X = all_exp_features
        self.y = all_label_dfs['Which Hand'].values
        self.all_y = all_label_dfs
        self.assertEqual(len(self.X), len(self.y))

        # Create param grid.
        self.param_grid = [
            {'classifier': [LogisticRegression()],
             'classifier__penalty': ['l2'],
             'classifier__C': [100, 80, 60, 40, 20, 15, 10, 8, 6, 4, 2, 1.0, 0.5, 0.1, 0.01],
             'classifier__solver': ['newton-cg', 'lbfgs', 'liblinear']},
            {'classifier': [RandomForestClassifier()],
             'classifier__bootstrap': [True, False],
             'classifier__max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
             'classifier__max_features': ['auto', 'sqrt'],
             'classifier__min_samples_leaf': [1, 2, 4],
             'classifier__min_samples_split': [2, 5, 10],
             'classifier__n_estimators': [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]},
            {'classifier': [sklearn.svm.SVC()],
             'classifier__C': [50, 40, 30, 20, 10, 8, 6, 4, 2, 1.0, 0.5, 0.1, 0.01],
             'classifier__kernel': ['poly', 'rbf', 'sigmoid'],
             'classifier__gamma': ['scale']},
            {'classifier': [RidgeClassifier()],
             'classifier__alpha': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]}
        ]

    def test_basic_make_preds(self):
        # test default model
        model = TC.ReachClassifier(model=LogisticRegression())
        model.fit(self.X, self.y)
        self.assertTrue(len(model.predict(self.X)) != 0)

    def test_hyperparam_tuning(self):
        classifier = TC.ReachClassifier()

        # Create first pipeline for base without reducing features.
        model = Pipeline(steps=[('standardscaler', StandardScaler()),
                                ('classifier', RandomForestClassifier())])

        X_train, X_val, y_train, y_val = classifier.partition(self.X, self.y)
        best_model, _, _ = classifier.hyperparameter_tuning(
            X_train, X_val, y_train, y_val, model, self.param_grid, fullGridSearch=False)
        self.model = best_model
        self.model.fit(X_train, y_train)
        _, test_score = classifier.evaluate(self.model, X_val, y_val)
        preds = self.model.predict(self.X)

        self.assertEqual(len(self.X), len(preds)), "Incorrect Num Predictions!"
        self.assertTrue(test_score > 0.5, f"Score is less than chance: {test_score}")

    def test_smote(self):
        # test class imbalancing
        classifier = TC.ReachClassifier(model=LogisticRegression())
        self.model = classifier.model
        X_res, y_res = classifier.adjust_class_imbalance(self.X, self.y)
        X_train, X_val, y_train, y_val = classifier.partition(X_res, y_res)
        self.model.fit(X_train, y_train)
        _, post_test_score = classifier.evaluate(self.model, X_val, y_val)

        #self.assertTrue(post_test_score > 0.5, f"Score is less than chance: {post_test_score}")
        self.assertTrue(X_res.shape[1] == self.X.shape[1], "Incorrect shape!")

    def test_feature_selection(self):
        classifier = TC.ReachClassifier(model=LogisticRegression())
        k = 3
        X_selected, fs = classifier.do_feature_selection(self.X, self.y, k=k)
        X_train, X_val, y_train, y_val = classifier.partition(X_selected, self.y)
        #classify
        model = LogisticRegression(solver='liblinear')
        model.fit(X_train, y_train)
        # evaluate the model
        yhat = model.predict(X_val)
        # evaluate predictions
        accuracy = accuracy_score(y_val, yhat)
        # print(accuracy)

        # test plotting
        classifier.plot_features(fs, self.X)

        self.assertTrue(X_train.shape[1] == k, f"Incorrect number of features! {X_train.shape[1]}")
        self.assertTrue(X_train.shape[0] == y_train.shape[0], "Incorrect Number of Trials!")

    def test_classify(self):
        # test train, validate, predict
        classifier = TC.ReachClassifier()
        # adjust class imbalance, partition, feature selection
        X_selected, y_res, fs = classifier.pre_classify(self.X, self.y)
        # train and validate X_train_selected, y_train,
        best_model, val_score = classifier.train_and_validate(X_selected, y_res, self.param_grid)
        # predict on all X
        preds = best_model.predict(X_selected)

        self.assertTrue(len(preds) == len(y_res))
        self.assertTrue(val_score >= 0.60, f"{val_score}")

class TestClassificationHierarchy(TestCase):
    """ Test Overview
        - test making predictions in hierarhcy
        - test saving trained models in hierarchy
        - test loading trained models in hierarchy

    """
    # Create param grid.
    param_grid = [
        {'classifier': [LogisticRegression()],
         'classifier__penalty': ['l2'],
         'classifier__C': [100, 80, 60, 40, 20, 15, 10, 8, 6, 4, 2, 1.0, 0.5, 0.1, 0.01],
         'classifier__solver': ['newton-cg', 'lbfgs', 'liblinear']},
        {'classifier': [RandomForestClassifier()],
         'classifier__bootstrap': [True, False],
         'classifier__max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
         'classifier__max_features': ['auto', 'sqrt'],
         'classifier__min_samples_leaf': [1, 2, 4],
         'classifier__min_samples_split': [2, 5, 10],
         'classifier__n_estimators': [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]},
        {'classifier': [sklearn.svm.SVC()],
         'classifier__C': [50, 40, 30, 20, 10, 8, 6, 4, 2, 1.0, 0.5, 0.1, 0.01],
         'classifier__kernel': ['poly', 'rbf', 'sigmoid'],
         'classifier__gamma': ['scale']},
        {'classifier': [RidgeClassifier()],
         'classifier__alpha': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]}
    ]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # LOAD DATA
        preprocessor = TC.Preprocessor()
        all_kin_features = preprocessor.load_data(f'{TC.folder_name}/kin_feat.pkl')
        all_exp_features = preprocessor.load_data(f'{TC.folder_name}/exp_feat.pkl')
        all_exp_features.dropna(axis=1, inplace=True)
        all_label_dfs = preprocessor.load_data(f'{TC.folder_name}/label_dfs.pkl')

        self.X = TC.ReachClassifier.mean_df(all_exp_features)
        self.y = all_label_dfs

    def test_hierarchy_train(self):
        # test training and saving models
        t = TC.ClassificationHierarchy()
        t.run_hierarchy(self.X, self.y, self.param_grid, models=None, save_models=True)

        # test loading and training
        #t = TC.ClassificationHierarchy()
        #models = [f'{TC.folder_name}/TrialTypeModel.joblib', f'{TC.folder_name}/NumReachesModel.joblib',
        #          f'{TC.folder_name}/WhichHandModel.joblib']
        #t.run_hierarchy(self.X, self.y, self.param_grid, models, save_models=False)

    def test_split(self):
        """
        # test classify and split
        classifier = TC.ReachClassifier()
        t = TC.ClassificationHierarchy()
        y_0 = [random.randint(0, 1) for _ in np.arange(len(self.y))]  # 1 if null, 0 if real trial
        model_0, val_score_0, preds_0 = t.classify(classifier, self.X, y_0, self.param_grid, False,
                                                      f'{TC.folder_name}/TrialTypeModel', None, True)

        # test split
        X_left, y_left, X_right, y_right = t.split(preds_0, self.X, self.y)

        self.assertEqual(len(X_left), len(y_left))
        self.assertEqual(len(X_right), len(y_right))
        self.assertEqual(len(X_left)+len(X_right), len(self.X))"""
        pass