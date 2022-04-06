from DataLoader import get_matched_labels_and_data as get_data
import pandas as pd
from sklearn.decomposition import PCA
import plotly.express as px
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf


class Visualize:

    def __init__(self):
        dataframe_address = ''
        labels_address = ''
        col_names = ['palm', 'wrist', 'forearm', 'index']  # segments to obtain kinematics from
        drop_list = [['13', 'S3', 20]]  # rat,date,session indexes to drop from data
        self.matched_data = get_data(dataframe_address, labels_address, col_names, drop_list)
        self.PCA = PCA()
        self.PCA_data = self.PCA.fit_transform(self.matched_data)

    def visualize_PCA_cumulative_sum(self):
        exp_var_cumul = np.cumsum(self.PCA_data.explained_variance_ratio_)
        fig = px.area(
            x=range(1, exp_var_cumul.shape[0] + 1),
            y=exp_var_cumul,
            labels={"x": "# Components", "y": "Explained Variance"}
        )
        fig.show()
