""" Class designed to in-take a trialized data object, perform supervised behavioral classification
 using pre-trained models, and return predicted class labels. Functions are included to start, stop,
 and utilize various pre-trained classification algorithms to predict class labels."""


import pandas as pd
import glob, os


class ReachPredict:
    def __init__(self, input_data):
        """ This function intializes class variables and starts each network. This function returns a binary prediction label with key
            ['', '', '', ' ']. """
        self.input_data = input_data # Array of variables to be classified.
        self.binary_key, self.binary_predictions = [], [] # fill in binary key later
        self.classifier_object, self.networks = [], []
        self.classifier_object_address = ''
        self.results = []
        self.reach_classifier,  self.tug_classifier, self.LR_BI_classifier, self.L_R_classifier, \
            self.low_vs_high_num_reaches, self.one_vs_two_or_three_reaches, \
            self.two_three_reaches_vs_bout = 0, 0, 0, 0, 0, 0, 0
        self.classifier_key = ['reach_no_reach', 'tug', 'LR_or_bi', 'L_or_R', 'low_high_reaches', 'low_class_split', 'high_class_split']
        self.fetch_pretrained_classifier_data()
        self.reshape_input_data()
        return

    def get_prediction_vector(self):
        for ix, pred_class in enumerate(self.classifier_key):
            self.networks.append(self.set_classifier_with_data(pred_class))
            self.results.append(self.use_network_generate_predictions(self.networks[ix]))
        return self.results

    def fetch_pretrained_classifier_data(self):
        """ Function to load and save paths as a list of objects."""
        os.chdir(self.classifier_object_address)
        for file in glob.glob(".txt"): # change to appropriate folder identifier
            self.classifier_object.append(file)
        return

    def set_classifier_with_data(self, classifier_key=''):
        """ Initializes a pre-trained classification algorithm, using the key to set the correct classifier identity (Emily).
        """
        return

    def reshape_input_data(self):
        """ Transform input data into correct shape for classification (Brett)"""
        return

    def use_network_generate_predictions(self, classifier):
        """ Function to generate single, binary prediction from named classifier w/ correct class key. """
        classification_prediction = 0
        return classification_prediction