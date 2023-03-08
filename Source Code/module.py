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

# importing libraries for the LSTM model that would be used
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from keras.layers import Input, Dense, LSTM
from keras.models import Model
from scikeras.wrappers import KerasClassifier


np.random.seed(10) #for consistency by ensuring same random values are generarted when needed

# read fasta files to list
def readFasta(input_file):
    fasta_sequences = SeqIO.parse(open(input_file), 'fasta')
    all_sequences = []
    for fasta in fasta_sequences:
        name, sequence = fasta.id, str(fasta.seq)
        one_sequence = [name, sequence]
        all_sequences.append(one_sequence)
    return all_sequences

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
def lstm_baseline_model(X_train):
    # create model
    model = Sequential()
    model.add(LSTM(100, input_shape=(X_train.shape[1], 1), activation='relu'))
    # model.add(Dense(X_train.shape[1], input_shape=(X_train.shape[1],), activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model