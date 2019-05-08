# -*- coding: utf-8 -*-
"""
Created on Wed May  8 07:57:19 2019

@author: piotr
"""

import os
import pandas as pd
import numpy as np
import seaborn as sns

from sklearn.cluster import KMeans


DATASET_PATH = 'datasets/'

data_path = os.path.join(DATASET_PATH, 'pima-indians-diabetes.csv')
dataset = pd.read_csv(data_path, header=None)

# Because thr CSV doesn't contain any header, we add column names 
# using the description from the original dataset website
dataset.columns = [
    "NumTimesPrg", "PlGlcConc", "BloodP",
    "SkinThick", "TwoHourSerIns", "BMI",
    "DiPedFunc", "Age", "HasDiabetes"]

#corr = dataset.corr()
#sns.heatmap(corr, annot = True)

df = dataset.drop('HasDiabetes', axis = 1)

# K-MEANS

km = KMeans(n_clusters = 3)

