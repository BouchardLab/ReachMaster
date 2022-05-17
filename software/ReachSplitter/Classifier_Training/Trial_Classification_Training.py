"""
    Written by Brett Nelson, UC Berkeley/ Lawrence Berkeley National Labs, NSDS Lab 12/8/2020
               Emily Nguyen, UC Berkeley

    This code is intended to create and implement structure supervised classification of coarsely
    segmented trial behavior from the ReachMaster experimental system.
    Functions are designed to work with a classifier of your choice.

    Edited: 12/8/2020
"""
import argparse
from imblearn.over_sampling import SMOTE  # for adjusting class imbalances
import sklearn
import joblib  # for saving sklearn models
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from PycharmProjects.ReachMaster.software.ReachSplitter.Classifier_Training import Classification_Utils as CU
import pandas as pd
import numpy as np
import h5py
# classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, RandomizedSearchCV, train_test_split, GridSearchCV
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn import preprocessing

# set global random seed #
np.random.seed(42)

def get_single_trial(df,date,session,rat):
    rr = df.loc[df['Date'] == date]
    rr = rr.loc[rr['S'] == session]
    new_df = rr.loc[rr['rat'] == rat]
    return new_df


def preprocessing_df(df):
    d=[]
    for index, row_value in df['Date'].iteritems():
        if "_" in row_value:
            d.append((row_value)[4:6])
        else:
            d.append((row_value)[5:7])
    df['Date']=d
    return df


def main_1_vec_labels(labels, key_names, save=False, ):
    """ Processes DLC trial labels.
    Args:
        labels (list of lists): list of unprocessed trial labels
        key_names (list of str): ordered file key names that must correspond to labels
        save (boolean): True to save, False (default) otherwise

    Returns:
        vectorized_labels (list of lists)
    """
    # vectorize DLC labels into ML ready format
    # function make_vectorized_labels returns
    #   (1) list: vectorized list of labels,
    #   (2) e: vectorized array of reading indices (unused variable)
    vectorized_labels = []
    for label in labels:
        vectorized_label, _ = CU.make_vectorized_labels(label)
        vectorized_labels.append(vectorized_label)

    # save each vectorized label
    if save:
        file_name = "vectorized_labels"
        hf = h5py.File(file_name, 'w')
        for i in np.arange(len(vectorized_labels)):
            hf.create_dataset(key_names[i], data=vectorized_labels[i])
        hf.close()
        print("Saved vectorized labels.")

    print("Finished vectorizing labels.")
    return vectorized_labels


def main_2_kin_exp_blocks(kin_data, exp_data, all_block_names, save=False):
    """ Gets blocks from kinematic and experimental data.
    Args:
        kin_data (list of str): path to kinematic data files
        exp_data (str): path to kinematic data file
        block_names (list of lists of str): for each rat, block keys corresponding to those in data
            shape num kin data files by num blocks to take from that data file by
        save (bool): True to save, False (default) otherwise

    Returns:
        kin_blocks (list of df)
        exp_blocks (list of df)
        kin_file_names (list of str)
        exp_file_names (list of str)
    """
    all_kin_blocks = []
    all_exp_blocks = []
    all_kin_file_names = []
    all_exp_file_names = []

    # for each rat
    for i in np.arange(len(kin_data)):
        # load kinematic and experimental data
        kin_df, exp_df = CU.load_kin_exp_data(kin_data[i], exp_data)
        block_names = all_block_names[i]

        # get blocks
        #   rat (str): rat ID
        #   date (str): block date in robot_df_
        #   kdate (str): block date in kin_df_
        #   session (str): block session
        kin_blocks = []
        exp_blocks = []
        for i in np.arange(len(block_names)):
            # get blocks
            rat, kdate, date, session = block_names[i]
            kin_block_df = CU.get_kinematic_block(kin_df, rat, kdate, session)
            exp_block_df = get_single_trial(exp_df, date, session, rat)

            # append blocks
            kin_blocks.append(kin_block_df)
            exp_blocks.append(exp_block_df)

        # save kinematic and experimental blocks
        kin_file_names = []
        exp_file_names = []
        for i in np.arange(len(block_names)):
            rat, kdate, date, session = block_names[i]
            kin_key = rat + kdate + session
            exp_key = rat + date + session
            kin_block_name = 'kin_block' + "_" + kin_key
            exp_block_name = 'exp_block' + "_" + exp_key
            kin_file_names.append(kin_block_name)
            exp_file_names.append(exp_block_name)
            if save:
                kin_blocks[i].to_pickle(kin_block_name)
                exp_blocks[i].to_pickle(exp_block_name)

        # append results
        all_kin_blocks = all_kin_blocks + kin_blocks
        all_exp_blocks = all_exp_blocks + exp_blocks
        all_kin_file_names = all_kin_file_names + kin_file_names
        all_exp_file_names = all_exp_file_names + exp_file_names

    if save:
        print("Saved kin & exp blocks.")
    print("Finished creating kin & exp blocks.")
    return all_kin_blocks, all_exp_blocks, all_kin_file_names, all_exp_file_names

# TODO find matching exp to kin block
# TODO change ML function below to new format

def main_3_ml_feat_labels(vectorized_labels, label_key_names,
                          kin_blocks, exp_blocks, kin_file_names, exp_file_names,
                          et, el, wv, window_length, pre,
                          load=False, save=False):
    """
    Returns ml feature and label arrays.
    Args:
        vectorized_labels: return value from main_1
        label_key_names: key names in 'vectorized_labels'
        kin_blocks: return value from main_2
        exp_blocks: return value from main_2
        kin_file_names: kin block file names
        exp_file_names: exp block file names
        et, el, wv, window_length, pre (int): 'make_s_f_trial_arrays_from_block' args
        load (bool): True to load file names, False (default) otherwise
        save (bool): True to save results, False (default) otherwise

    Returns:
        final_ML_feature_array, final_labels_array, feat_names

    Notes:
        labels and blocks must match!
    """
    # init data handling variables
    if load:
        vectorized_labels = []
    c_positions = []
    c_probabilities = []

    # for each block
    assert (len(label_key_names) == len(kin_file_names))
    num_blocks = len(label_key_names)
    for i in np.arange(num_blocks):
        # load vectorized labels, kin block, and exp block data
        if load:
            block_label = CU.load_hdf("vectorized_labels", label_key_names[i])
            kin_block_df = pd.read_pickle(kin_file_names[i])
            exp_block_df = pd.read_pickle(exp_file_names[i])
            vectorized_labels.append(block_label)
        else:
            block_label = vectorized_labels[i]
            kin_block_df = kin_blocks[i]
            exp_block_df = exp_blocks[i]

        # trial-ize data
        hot_vector, trialized_kin_data, feat_names, exp_data \
            = CU.make_s_f_trial_arrays_from_block(kin_block_df, exp_block_df, et, el, wv, window_length, pre)

        # Match with label
        matched_kin_data, matched_exp_data = CU.match_stamps(trialized_kin_data, block_label, exp_data)

        # match kin and exp features
        # create_ML_array args: matched kin array, matched ez array
        c_pos, c_prob = CU.create_ML_array(matched_kin_data, matched_exp_data)

        # append results
        c_positions.append(c_pos)
        c_probabilities.append(c_prob)

    # resize data
    final_ML_feature_array_XYZ, final_labels_array \
        = CU.stack_ML_arrays(c_positions, vectorized_labels)
    final_ML_feature_array_prob, _ \
        = CU.stack_ML_arrays(c_probabilities, vectorized_labels)

    # concat horizontally XYZ and prob123 ml feature arrays
    # (total num labeled trials x (3*num kin feat)*2 +num exp feat = 174 for XYZ and prob, window_length+pre)
    final_ML_feature_array = np.concatenate((final_ML_feature_array_XYZ, final_ML_feature_array_prob),
                                            axis=1)  # this causes issues with norm/zscore

    if save:
        # Save final_ML_array and final_feature_array in h5 file
        with h5py.File('ml_array_RM16.h5', 'w') as hf:
            hf.create_dataset("RM16_features", data=final_ML_feature_array)
            hf.create_dataset("RM16_labels", data=final_labels_array)
        print("Saved final ml feat and label arrays.")
        with open('feat_names.npy', 'wb') as f:
            np.save(f, feat_names)

    print("Finished creating final ML feat and labels.")
    return final_ML_feature_array, final_labels_array, feat_names


########################
# Classification
########################


def classify(model, X, Y, k):
    """
    Classifies trials as null vs not null.
    Args:
        model: sklearn model
        X_train (array): features, shape (num trials, num feat*num frames)
        hand_labels_y_train (array): labels shape (num trials)
        k (int): number of kfolds for cross validation score

    Returns:
        classifier_pipeline (pipeline): trained model
        predictions (array): classifier trial predictions
        score (int): mean of cross validation scores
    """
    # create pipeline
    classifier_pipeline = make_pipeline(preprocessing.StandardScaler(), model)

    # fit to training data
    classifier_pipeline.fit(X, Y)

    # calculate mean of kfold cv
    score = np.mean(cross_val_score(classifier_pipeline, X, Y, cv=k))

    # predict X_train data
    predictions = classifier_pipeline.predict(X)

    return classifier_pipeline, predictions, score


def train_and_save_model(model, param_grid, X_train, y_train, feat_names, num_frames, file_name, save=False):
    """
    Args:
        X_train:
        y_train:
        feat_names:
        num_frames:

    Returns:

    Reference: https://towardsdatascience.com/logistic-regression-model-tuning-with-scikit-learn-part-1-425142e01af5
    """

    # partition into validation set
    X_train_paritioned, X_val, y_train_paritioned, y_val = train_test_split(X_train, y_train, test_size=0.2)
    # adjust for class imbalance
    #   oversamples the minority class by synthetically generating additional samples
    sm = SMOTE(random_state=42)
    X_train_res, y_train_res = sm.fit_resample(X_train_paritioned, y_train_paritioned)

    # 1b. hyperparameter tuning
    best_grid, best_params_, tuned_accuracy = \
        hyperparameter_tuning(model, param_grid, X_train_res, y_train_res, X_val, y_val,
                              fullGridSearch=False)

    if save:
        joblib.dump(best_grid, file_name+'.joblib')

        # classify all training data
    classifier_pipeline_null, predictions_null, cv_score_null = classify(model, X_train, y_train, k=3)

    return classifier_pipeline_null, predictions_null, cv_score_null


def evaluate(model, test_features, test_labels):
    predictions = model.predict(test_features)
    accuracy = accuracy_score(test_labels, predictions)
    print('Accuracy', accuracy)
    return accuracy


def hyperparameter_tuning(model, param_grid, train_features, train_labels, val_features, val_labels,
                          fullGridSearch=False, save=False):
    """
    Performs hyperparameter tuning and returns best trained model.
    Args:
        model:
        param_grid:
        train_features:
        train_labels:
        val_features:
        val_labels:
        fullGridSearch: True to run exhaustive param search, False runs RandomizedSearchCV
        save (bool): True to save model, False otherwise

    Returns:
        tuned model
        parameters found through search
        accuracy of tuned model

    Reference: https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74
    """
    # Use the random grid to search for best hyperparameters
    if fullGridSearch:
        # Instantiate the grid search model
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid,
                                   cv=3, n_jobs=-1, verbose=2)
    else:
        # Random search of parameters, using 3 fold cross validation,
        # search across 100 different combinations, and use all available cores
        grid_search = RandomizedSearchCV(estimator=model, param_distributions=param_grid, n_iter=2, cv=5,
                                         random_state=42, verbose=2, n_jobs=-1)

    # Fit the random search model
    grid_search.fit(train_features, train_labels)

    base_model = RandomForestClassifier()
    base_model.fit(train_features, train_labels)
    base_accuracy = evaluate(base_model, val_features, val_labels)

    best_grid = grid_search.best_estimator_
    tuned_accuracy = evaluate(best_grid, val_features, val_labels)

    print('Improvement % of', (100 * (tuned_accuracy - base_accuracy) / base_accuracy))

    return best_grid, grid_search.best_params_, tuned_accuracy


def main_4_classify(final_ML_feature_array, final_labels_array, feat_names, load=False, save=False):
    if load:
        # Load final_ML_array and final_feature_array in h5 file
        with h5py.File('ml_array_RM16.h5', 'r') as f:
            final_ML_feature_array = f['RM16_features'][:]
            final_labels_array = f['RM16_labels'][:]
        with open('feat_names.npy', 'rb') as f:
            feat_names = np.load(f)
        feat_names = [str(t[0]) for t in feat_names]  # un-nest

    ### prepare data ###

    # reshape features to be (num trials, num feat * num frames)
    num_frames = final_ML_feature_array.shape[2]
    final_ML_feature_array = final_ML_feature_array.reshape(final_ML_feature_array.shape[0],
                                                            final_ML_feature_array.shape[1] *
                                                            final_ML_feature_array.shape[2])

    # partition data into test, train, and validation sets
    X_train, X_test, y_train, y_test = train_test_split(final_ML_feature_array, final_labels_array,
                                                        test_size=0.2)
    # type_labels_y_train, num_labels_y_train, hand_labels_y_train, tug_labels_y_train, switch_labels_y_train
    y_train = CU.get_ML_labels(y_train)
    y_test = CU.get_ML_labels(y_test)

    # Create first pipeline for base without reducing features.
    model = Pipeline(steps=[('standardscaler', StandardScaler()),
                            ('classifier', RandomForestClassifier())])

    # Create param grid.
    # TODO expand search
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

    # 1a. feature selection
    # TODO adjust
    keywords = ["Nose", "Palm"]
    feat_df = CU.reshape_final_ML_array_to_df(num_frames, X_train, feat_names)
    _, X_train_selected = CU.select_feat_by_keyword(feat_df, keywords)

    ### classify null trials ###
    type_labels_y_train = y_train[0]
    classifier_pipeline_null, predictions_null, score_null =\
        train_and_save_model(model, param_grid, X_train_selected, type_labels_y_train, feat_names, num_frames,
                             "null_classifier", save=True)

    # 1c. REMOVE NULL TRIALS
    toRemove = 1  # remove null trials # 1 if null, 0 if real trial
    print(np.array(X_train).shape, np.array(y_train).shape)
    X_train_null, y_train_null = CU.remove_trials(X_train, y_train, predictions_null, toRemove)
    print(X_train_null.shape, y_train_null.shape)

    # 2. NUM REACHES

    # 2a. feature selection
    # TODO adjust
    keywords = ['Handle', 'Palm']  # ["'Handle', 'X'", "'Palm', 'X'"]
    feat_df = CU.reshape_final_ML_array_to_df(num_frames, X_train_null, feat_names)
    _, X_train_selected = CU.select_feat_by_keyword(feat_df, keywords)

    # 2b. classify
    num_labels_y_train = y_train_null[1]
    classifier_pipeline_reaches, predictions_reaches, score_reaches = \
        train_and_save_model(model, param_grid, X_train_selected, num_labels_y_train, feat_names, num_frames,
                             "num_reaches_classifier", save=True)

    # 2c. REMOVE >1 REACH TRIALS
    toRemove = 1  # remove >1 reaches # 0 if <1, 1 if > 1 reaches
    X_train_reaches, y_train_reaches = CU.remove_trials(X_train_null, y_train_null, predictions_reaches, toRemove)
    print(X_train_reaches.shape, y_train_reaches.shape)

    # 3. WHICH HAND

    # 3a. feature selection
    # TODO adjust
    keywords = ['Robot', 'Palm']
    feat_df = CU.reshape_final_ML_array_to_df(num_frames, X_train_reaches, feat_names)
    _, X_train_selected = CU.select_feat_by_keyword(feat_df, keywords)

    # 3b. classify
    hand_labels_y_train = y_train_reaches[2]
    classifier_pipeline_hand, predictions_hand, score_hand = \
        train_and_save_model(model, param_grid, X_train_selected, hand_labels_y_train, feat_names, num_frames,
                             "which_hand_classifier", save=True)

    # 3c. REMOVE lra/rla/bi HAND TRIALS
    toRemove = 1  # remove lra/rla/bi reaches # 1 if lra/rla/bi, 0 l/r reaches
    X_train_hand, y_train_hand = CU.remove_trials(X_train_reaches, y_train_reaches, predictions_hand, toRemove)
    print(X_train_hand.shape, y_train_hand.shape)

    #clf = joblib.load('which_hand_classifier.joblib')
    #print(clf)

    print(score_null, score_hand, score_reaches)
    print("Finished classification.")

    return [predictions_null, predictions_reaches, predictions_hand]  # TODO fix mismatched dim

def results_to_csv(results, reachIDs, file_name):
    """
    Saves classification predictions to CSV in current working directory.
    :param results: data to save
    :param file_name: for unique csv file naming
    :param reachIDs: predicted reach index start times
    :return: none
    """
    df = pd.DataFrame({'Null Preds': results[0],
                       'Num Reach Preds': results[1],
                       'Which Hand Preds': results[2],
                       'Reach ID Preds': reachIDs})
    #df.index += 1  # Ensures that the index starts at 1.
    df.to_csv(file_name + '.csv', index_label='Trial Num')


def main_5_save_results():
    # TODO fix
    # gen dummy results for e.g # RM16, 9-17, S2, 37 trials total, 1-indexed
    numTrials = 37
    results = np.random.randint(low=-1, high=2, size=(3, numTrials))  # high exclusive # if no classification, preds=-1
    reachIDs = np.random.randint(low=0, high=10000, size=numTrials)  # predicted start indices

    file_name = 'trialClassification'
    results_to_csv(results, reachIDs, file_name)


#######################
# MAIN
#######################
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--function", "-f", type=int, default=1, help="Specify which function to run")
    args = parser.parse_args()

    # labels
    labels = [CU.rm16_9_20_s3_label,
              CU.rm16_9_19_s3_label,
              CU.rm16_9_17_s2_label,
              CU.rm16_9_17_s1_label,
              CU.rm16_9_18_s1_label,

              # CU.rm15_9_25_s3_label,
              # CU.rm15_9_17_s4_label

              CU.rm14_9_20_s1_label,
              CU.rm14_9_18_s2_label
              ]
    label_key_names = ['rm16_9_20_s3_label',
                       'rm16_9_19_s3_label',
                       'rm16_9_17_s2_label',
                       'rm16_9_17_s1_label',
                       'rm16_9_18_s1_label',

                       'rm14_9_20_s1_label',
                       'rm14_9_18_s2_label'
                       ]
    # blocks
    kin_data_path = ['tkd16.pkl', 'tkd14.pkl']
    exp_data_path = 'experimental_data.pickle'
    block_names = [
        [['RM16', '0190920', '0190920', 'S3'],
         ['RM16', '0190919', '0190919', 'S3'],
         ['RM16', '0190917', '0190917', 'S2'],
         ['RM16', '0190917', '0190917', 'S1'],
         ['RM16', '0190918', '0190918', 'S1']],

        [['RM14', '0190920', '0190920', 'S1'],
         ['RM14', '0190918', '0190918', 'S2']]
    ]

    kin_file_names = ['kin_block_RM160190920S3',
                      'kin_block_RM160190919S3',
                      'kin_block_RM160190917S2',
                      'kin_block_RM160190917S1',
                      'kin_block_RM160190918S1',

                      'kin_block_RM140190920S1',
                      'kin_block_RM140190918S2']

    exp_file_names = ['exp_block_RM160190920S3',
                      'exp_block_RM160190919S3',
                      'exp_block_RM160190917S2',
                      'exp_block_RM160190917S1',
                      'exp_block_RM160190918S1',

                      'exp_block_RM140190920S1',
                      'exp_block_RM140190918S2']

    # define params for trializing blocks
    et = 0
    el = 0
    wv = 5
    window_length = 4  # TODO change to preferences, default = 250
    pre = 2  # TODO change to preferences, default = 10

    if args.function == 1:
        vectorized_labels = main_1_vec_labels(labels, label_key_names, save=True)

    elif args.function == 2:
        kin_blocks, exp_blocks, kin_file_names, exp_file_names = \
            main_2_kin_exp_blocks(kin_data_path, exp_data_path, block_names, save=True)

    elif args.function == 3:
        # MUST DELETE ALL OLD DATA FILES BEFORE RUNNING if NOT using default args
        # TODO make an os assert for this or fix
        final_ML_feature_array, final_labels_array, feat_names = \
            main_3_ml_feat_labels([], label_key_names,
                                  [], [], kin_file_names, exp_file_names,
                                  et, el, wv, window_length, pre,
                                  load=True, save=True)
    elif args.function == 4:
        main_4_classify([], [], [], load=True, save=True)
        # TODO given rat or block, return classification labels (no shuffling!)
    elif args.function == 5:
        main_5_save_results()
    elif args.function == 6:
        # runs all without saving files into current working directory
        vectorized_labels = main_1_vec_labels(labels, label_key_names, save=False)
        kin_blocks, exp_blocks, kin_file_names, exp_file_names = main_2_kin_exp_blocks(kin_data_path, exp_data_path,
                                                                                       block_names, save=False)
        final_ML_feature_array, final_labels_array, feat_names = \
            main_3_ml_feat_labels(vectorized_labels, label_key_names,
                                  kin_blocks, exp_blocks, kin_file_names, exp_file_names,
                                  et, el, wv, window_length, pre,
                                  load=False, save=False)
        main_4_classify(final_ML_feature_array, final_labels_array, feat_names, load=False, save=False)

    else:
        raise ValueError("Cannot find specified function number")
