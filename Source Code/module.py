# import python packages
import pandas as pd # for reading data from csv
import numpy as np # for handling multi-dimensional array operation
from sklearn.model_selection import train_test_split # for spliting dataset
from sklearn.preprocessing import MinMaxScaler  # for normalization

from sklearn.ensemble import RandomForestClassifier # classifier for RF classification
from sklearn.svm import SVC  # classifier for SVM classification
from sklearn.neural_network import MLPClassifier # classifier for DNN classification
from sklearn.naive_bayes import GaussianNB # classifier for Naive Bayes
from sklearn.linear_model import LogisticRegression # classifier for LR classification
from deepforest import CascadeForestClassifier # classifier for deepforest classification

from sklearn.model_selection import cross_val_score # for cross validation
from sklearn import metrics # for evaluation
from sklearn.metrics import make_scorer # for evaluation
import time # for measuring time
import os, psutil; # for measuring memory usage
import xlsxwriter
from datetime import datetime
import os # for running python commands
import time
import mimetypes # for MIME types checking
from Bio import SeqIO
from random import sample
import sys
import shutil
import matplotlib.pyplot as plt #for plotting graphs
import seaborn as sns # for colorful 2d plotting
from sklearn.inspection import permutation_importance
from sklearn.decomposition import PCA # for PCA feature extraction
from sklearn.manifold import TSNE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import FastICA
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_validate, KFold
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, roc_auc_score
from sklearn.tree import DecisionTreeClassifier
import tracemalloc
from sklearn.feature_selection import mutual_info_classif
import joblib
from sklearn.model_selection import StratifiedGroupKFold

# importing libraries for the LSTM model that would be used
from keras.models import Sequential, Model
from keras.layers import Input, Dense, LSTM
from scikeras.wrappers import KerasClassifier

from sklearn.utils import resample
from scipy.stats import sem, t


seed = 10
np.random.seed(seed) #for consistency by ensuring same random values are generarted when needed

# read fasta files to list
def readFasta(input_file):
    fasta_sequences = SeqIO.parse(open(input_file), 'fasta')
    all_sequences = []
    for fasta in fasta_sequences:
        name, sequence = fasta.id, str(fasta.seq)
        one_sequence = [name, sequence]
        all_sequences.append(one_sequence)
    return all_sequences

# group protein sequences
from collections import defaultdict
def groupProteinSequences(sequences, start_group_id):
    groups = defaultdict(list)
    group_ids = []

    for name, sequence in sequences:
        groups[sequence].append(name)

    for group_id, (_, sequences) in enumerate(groups.items(), start=start_group_id):
        group_ids.extend([group_id] * len(sequences))

    return groups, group_ids

# validate protein sequence
def protcheck(seq):
    allowedAA = "ARNDCQEGHILKMFPSTWYV"
    if len(seq) > 2 and all(c in allowedAA for c in seq): return True
    else: return False

# delete invalid protein rows
def deleteInvalidProt(host_prot, pathogen_prot, intlabel):
    itr = 1
    while True:
        index_list = []
        total = 0
        for i in range(0, len(host_prot)):
            if (protcheck(host_prot[i][1]) == False or protcheck(pathogen_prot[i][1]) == False):
                index_list.append(i)
                total += 1
        print("Rows deleted in "+str(intlabel)+" protein (Itr "+str(itr)+"): ", total)
        for index in sorted(index_list, reverse=True):
            del host_prot[index]
            del pathogen_prot[index]
        if (total == 0): break
        itr += 1
    return host_prot, pathogen_prot

# converts protein sequence to feature vectors
def generateFeatures(input_path, output_path):
    # os.system("python iFeature/iFeature.py --file "+input_path+" --type CKSAAP --out "+output_path)
    # 0 in the code below shows that the composition of k-spaced AAP is with k=0
    os.system("python iFeature/codes/CKSAAP.py "+input_path+" 0 "+output_path)
    # os.system("python iFeature/iFeature.py --file "+input_path+" --type AAC --out "+output_path)
    return output_path

# converts protein sequence from list to fasta format
def convertToFasta(seq_list, output_path):
    ofile = open(output_path, "w")
    for i in range(len(seq_list)):
        ofile.write(">" + seq_list[i][0] + "\n" +seq_list[i][1] + "\n")
    ofile.close()
    return output_path

# create directory
def createDir(dir_path):
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    return dir_path

# delete directory
def deleteDir(dir_path):
    try:
        shutil.rmtree(dir_path)
    except OSError as e:
        print("Error: %s - %s." % (e.filename, e.strerror))

# creating baseline model for the LSTM deep neural network model with 1 hidden layer of 100neurons
def lstm_baseline_model_old(X_train):
    # create model
    model = Sequential()
    model.add(LSTM(100, input_shape=(X_train.shape[1], 1), activation='relu'))
    # model.add(Dense(X_train.shape[1], input_shape=(X_train.shape[1],), activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def lstm_baseline_model(dim):
    # Define the model architecture
    model = Sequential()
    model.add(LSTM(units=dim, activation='relu', return_sequences=True, input_shape=(dim, 1)))
    model.add(LSTM(units=dim * 0.8, activation='relu', return_sequences=True))
    model.add(LSTM(units=dim * 0.6, activation='relu', return_sequences=True))
    model.add(LSTM(units=dim * 0.4, activation='relu', return_sequences=True))
    model.add(LSTM(units=dim * 0.2, activation='relu', return_sequences=True))
    model.add(LSTM(units=dim * 0.1, activation='relu', return_sequences=False))
    model.add(Dense(units=1, activation='sigmoid'))

    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def custom_scorer(y_true, y_pred, custom_metric):
    if custom_metric == "accuracy":
        return metrics.accuracy_score(y_true, y_pred)
    elif custom_metric == "sensitivity":
        return metrics.recall_score(y_true, y_pred)
    elif custom_metric == "specificity":
        return metrics.recall_score(y_true, y_pred, pos_label=0)
    elif custom_metric == "precision":
        return metrics.precision_score(y_true, y_pred)
    elif custom_metric == "f1":
        return metrics.f1_score(y_true, y_pred)
    elif custom_metric == "mcc":
        return metrics.matthews_corrcoef(y_true, y_pred)
    elif custom_metric == "auroc":
        return metrics.roc_auc_score(y_true, y_pred)
    else:
        return

def multicollinearity(X):
    corr_matrix = X.corr().abs()

    # Select the upper triangle of the correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    # Find features with high correlation (threshold can be adjusted based on the need)
    to_drop = [column for column in upper.columns if any(upper[column] > 0.8)]

    # Create the new dataset with the selected features
    X_mc = X.drop(to_drop, axis=1)

    # Print the list of dropped features
    print("Features dropped due to multicollinearity: ", to_drop)
    return X_mc

def information_gain(X, y):
    # Calculate the mutual information between each feature and the target
    ig_scores = mutual_info_classif(X, y)
    return ig_scores

    # # Sort the features by their information gain score in descending order
    # sorted_indices = sorted(range(len(ig_scores)), key=lambda i: ig_scores[i], reverse=True)
    #
    # # Return the top 1000 feature indices and their corresponding information gain scores
    # top_indices = sorted_indices[:1000]
    # top_scores = ig_scores[top_indices]
    #
    # return top_indices, top_scores


# def information_gain(X, y):
#     # Calculate the mutual information between each feature and the target
#     ig_scores = mutual_info_classif(X, y)
#
#     # Sort the features by their information gain score in descending order
#     sorted_indices = sorted(range(len(ig_scores)), key=lambda i: ig_scores[i], reverse=True)
#
#     # Return the top 1000 feature indices and their corresponding information gain scores
#     top_indices = sorted_indices[:1000]
#     top_scores = ig_scores[top_indices]
#
#     return top_indices, top_scores

# from statsmodels.stats.outliers_influence import variance_inflation_factor
# def calc_vif(X):
#     # Add a small constant value to the diagonal of the correlation matrix to avoid divide-by-zero errors
#     corr = X.corr() + 0.01 * np.identity(X.shape[1])
#
#     # Calculate the VIF scores
#     vif = [variance_inflation_factor(corr.values, i) for i in range(corr.shape[0])]
#
#     # Select the features with VIF scores below a threshold (number can be adjusted based on the need)
#     selected_features = X.columns[vif < 5]
#
#     # Create the new dataset with the selected features and target variable
#     return X[selected_features]

# works for python version >3.10 and I currently have 3.9
# def custom_scorer(y_true, y_pred, custom_metric):
#     match custom_metric:
#         case "accuracy":
#             return metrics.accuracy_score(y_true, y_pred)
#         case "sensitivity":
#             return metrics.recall_score(y_true, y_pred)
#         case "specificity":
#             return metrics.recall_score(y_true, y_pred, pos_label=0)
#         case "precision":
#             return metrics.precision_score(y_true, y_pred)
#         case "f1":
#             return metrics.f1_score(y_true, y_pred)
#         case "mcc":
#             return metrics.matthews_corrcoef(y_true, y_pred)
#         case "auroc":
#             return metrics.roc_auc_score(y_true, y_pred)
#         case _:
#             return