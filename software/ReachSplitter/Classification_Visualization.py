"""
    Written by Brett Nelson, UC Berkeley/ Lawrence Berkeley National Labs, NSDS Lab 12/8/2020
    Functions to visualize classifier and segmentation model results using the yellowbrick library.
    Functions are designed to work with a classifier of your choice.

    Edited: 12/14/2020

"""

from sklearn.metrics import f1_score
from yellowbrick.classifier import ClassificationReport, DiscriminationThreshold, ConfusionMatrix, ClassBalance, ClassPredictionError
import numpy as np
from sklearn import metrics


def visualize_model(X, y, estimators,pred=False,disc=False, conf=False, bal=False,**kwargs):
    """
    Visualize models using the yellowbrick plotting library.
    """

    # Instantiate the classification model and visualizer
    visualizer = ClassificationReport(
        estimators, classes=['Reach, 1 Reach, or L/R Reach', 'Null, Multiple Reaches, Or Multiple Arms'],
        cmap="YlGn", size=(600, 360), **kwargs
    )
    visualizer.fit(X, y)
    visualizer.score(X, y)
    visualizer.show()
    if pred:
        class_prediction_errors(X, y, estimators, **kwargs)
    if disc:
        discrimination_thresholding(X, y, estimators, **kwargs)
    if conf:
        confusion_matrix(X, y, estimators, **kwargs)
    if bal:
        plot_class_balance(y, **kwargs)


def class_prediction_errors(xx,yy,estimatorss,**kwargs):
    vz2 = ClassPredictionError(estimatorss, classes=['Reach, 1 Reach, or L/R Reach', 'Null, Multiple Reaches, Or Multiple Arms'],
        cmap="YlGn", size=(600, 360), **kwargs)
    vz2.fit(xx, yy)
    vz2.score(xx, yy)
    vz2.show()


def discrimination_thresholding(xx,yy,estimatorss,**kwargs):
    vz = DiscriminationThreshold(estimatorss, classes=['Reach, 1 Reach, or L/R Reach', 'Null, Multiple Reaches, Or Multiple Arms'],
        cmap="YlGn", size=(600, 360), **kwargs)
    vz.fit(xx,yy)
    vz.score(xx,yy)
    vz.show()


def confusion_matrix(xx,yy,estimatorss,**kwargs):
    vz1 = ConfusionMatrix(estimatorss, classes=['Reach, 1 Reach, or L/R Reach', 'Discard'],
        cmap="YlGn", size=(600, 360), **kwargs)
    vz1.fit(xx, yy)
    vz1.score(xx, yy)
    vz1.show()


def plot_class_balance(yest,ttsplit=False):
    visualizer = ClassBalance(labels=["Reach/1R/LR", "Discard"])
    if ttsplit:
        visualizer.fit(yest, ttsplit) # observe inside of split
    else:
        visualizer.fit(yest)
    visualizer.show()


def score_model(X, y,estimator, **kwargs):
    """
    Test various estimators.
    """
    # Instantiate the classification model and visualizer
    estimator.fit(X, y, **kwargs)

    expected = y
    predicted = estimator.predict(X)
    # Compute and return F1 (harmonic mean of precision and recall)
    print("{}: {}".format(estimator.__class__.__name__, f1_score(expected, predicted)))


def visualize_models(xt,yt,estimators):
    for model in estimators:
        score_model(xt, yt, model)
        visualize_model(xt, yt, model)



def plot_decision_tree(estimator,X_test):
    n_nodes = estimator.tree_.node_count
    children_left = estimator.tree_.children_left
    children_right = estimator.tree_.children_right
    feature = estimator.tree_.feature
    threshold = estimator.tree_.threshold

    # The tree structure can be traversed to compute various properties such
    # as the depth of each node and whether or not it is a leaf.
    node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
    is_leaves = np.zeros(shape=n_nodes, dtype=bool)
    stack = [(0, -1)]  # seed is the root node id and its parent depth
    while len(stack) > 0:
        node_id, parent_depth = stack.pop()
        node_depth[node_id] = parent_depth + 1

        # If we have a test node
        if (children_left[node_id] != children_right[node_id]):
            stack.append((children_left[node_id], parent_depth + 1))
            stack.append((children_right[node_id], parent_depth + 1))
        else:
            is_leaves[node_id] = True

    print("The binary tree structure has %s nodes and has "
          "the following tree structure:"
          % n_nodes)
    for i in range(n_nodes):
        if is_leaves[i]:
            print("%snode=%s leaf node." % (node_depth[i] * "\t", i))
        else:
            print("%snode=%s test node: go to node %s if X[:, %s] <= %s else to "
                  "node %s."
                  % (node_depth[i] * "\t",
                     i,
                     children_left[i],
                     feature[i],
                     threshold[i],
                     children_right[i],
                     ))
    print()

    # First let's retrieve the decision path of each sample. The decision_path
    # method allows to retrieve the node indicator functions. A non zero element of
    # indicator matrix at the position (i, j) indicates that the sample i goes
    # through the node j.

    node_indicator = estimator.decision_path(X_test)

    # Similarly, we can also have the leaves ids reached by each sample.

    leave_id = estimator.apply(X_test)

    # Now, it's possible to get the tests that were used to predict a sample or
    # a group of samples. First, let's make it for the sample.

    sample_id = 0
    node_index = node_indicator.indices[node_indicator.indptr[sample_id]:
                                        node_indicator.indptr[sample_id + 1]]

    print('Rules used to predict sample %s: ' % sample_id)
    for node_id in node_index:
        if leave_id[sample_id] == node_id:
            continue

        if (X_test[sample_id, feature[node_id]] <= threshold[node_id]):
            threshold_sign = "<="
        else:
            threshold_sign = ">"

        print("decision id node %s : (X_test[%s, %s] (= %s) %s %s)"
              % (node_id,
                 sample_id,
                 feature[node_id],
                 X_test[sample_id, feature[node_id]],
                 threshold_sign,
                 threshold[node_id]))

    # For a group of samples, we have the following common node.
    sample_ids = [0, 1]
    common_nodes = (node_indicator.toarray()[sample_ids].sum(axis=0) ==
                    len(sample_ids))

    common_node_id = np.arange(n_nodes)[common_nodes]

    print("\nThe following samples %s share the node %s in the tree"
          % (sample_ids, common_node_id))
    print("It is %s %% of all nodes." % (100 * len(common_node_id) / n_nodes,))
    return


def print_preds(predictions_, y_lab):
    for i, m in enumerate(predictions_):
        m=m[0] # get pred entry
        if i ==0:
            print('Reaching Type (Null or Reach?) ')
        elif i ==1:
            print('Num of Reaches <2 or > 2?')
        elif i==2:
            print('Which Hand (LR or other) ?')
        elif i ==3:
            print('Tug of War?')
        elif i == 4:
            print('Arm Switching?')
        print("Accuracy:",metrics.accuracy_score(y_lab[i],m))




