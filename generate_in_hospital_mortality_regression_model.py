#this file contains the methods used to run the regression model to predict in hopsital mortality

from numpy.lib.function_base import append
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import auc
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier

import os
import numpy as np
import argparse
import pandas as pd
from tqdm import tqdm
from metrics import *
from matplotlib import pyplot
from prepare_features_mortality_pred import *

import re
import json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--subjects_root_path', type=str, help='Directory containing subject subdirectories.')
    parser.add_argument('-use_generated_features_file', action='store_true', help='Used to specify if the previously generated features in root/result should be used.')
    args = parser.parse_args()

    if args.use_generated_features_file:
        features = pd.read_csv(os.path.join(args.subjects_root_path, 'result\\features.csv'))
        outcomes = pd.read_csv(os.path.join(args.subjects_root_path, 'result\\outcomes.csv'))

        data_X = features.values
        data_Y = outcomes.values
    else:
        #read all subject folders
        (data_X, data_Y) = scale_values(args.subjects_root_path)

        #save X and Y data to 'result' map
        features = pd.DataFrame(data_X)
        file_name = 'features.csv'
        features.to_csv(os.path.join(args.subjects_root_path, 'result\\', file_name), index=False)

        outcomes = pd.DataFrame(data_Y)
        file_name = 'outcomes.csv'
        outcomes.to_csv(os.path.join(args.subjects_root_path, 'result\\', file_name), index=False)

        #read all subjects folders and save a summary file
        #(data_X, data_Y) = scale_values_alt(args.subjects_root_path)

        #read an already existing summary file
        #(data_X, data_Y) = read_values(args.subjects_root_path)


    trainX, testX, trainy, testy = train_test_split(data_X, data_Y, test_size=0.15, random_state=2)
    testy = np.array(testy)
    trainy = np.array(trainy)
    model = LogisticRegression(penalty='l2', C=0.1)#, random_state=42 #1 seem to be best for whole dataset
    #model = RandomForestClassifier(n_estimators= 1000, random_state= 42)
    model.fit(trainX, trainy)
    print('Test accuracy:'+ str(model.score(testX, testy)))



    ns_probs = [0 for _ in range(len(testy))]
    lr_probs = model.predict_proba(testX)
    lr_probs = lr_probs[:, 1]


    #confusion matrix for training data
    cm = confusion_matrix(trainy, model.predict(trainX))
    print(cm)
    cm = confusion_matrix(testy, model.predict(testX))
    print(cm)
    cm_display = ConfusionMatrixDisplay(cm).plot()
    pyplot.show()

    # calculate scores
    ns_auc = roc_auc_score(testy, ns_probs)
    lr_auc = roc_auc_score(testy, lr_probs)
    # summarize scores
    print('No Skill: ROC AUC=%.3f' % (ns_auc))
    print('Logistic: ROC AUC=%.3f' % (lr_auc))
    # calculate roc curves
    ns_fpr, ns_tpr, _ = roc_curve(testy, ns_probs)
    lr_fpr, lr_tpr, _ = roc_curve(testy, lr_probs)
    # plot the roc curve for the model
    pyplot.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
    pyplot.plot(lr_fpr, lr_tpr, marker='.', label='Logistic')
    # axis labels
    pyplot.xlabel('False Positive Rate')
    pyplot.ylabel('True Positive Rate')
    # show the legend
    pyplot.legend()
    # show the plot
    pyplot.show()

    # predict probabilities
    lr_probs = model.predict_proba(testX)
    # keep probabilities for the positive outcome only
    lr_probs = lr_probs[:, 1]
    # predict class values
    yhat = model.predict(testX)
    lr_precision, lr_recall, _ = precision_recall_curve(testy, lr_probs)
    lr_f1, lr_auc = f1_score(testy, yhat), auc(lr_recall, lr_precision)
    # summarize scores
    print('Logistic: f1=%.3f auc=%.3f' % (lr_f1, lr_auc))
    # plot the precision-recall curves
    positive_len = len(testy[testy==1])
    total_len = len(testy)
    no_skill = positive_len / total_len 
    pyplot.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
    pyplot.plot(lr_recall, lr_precision, marker='.', label='Logistic')
    # axis labels
    pyplot.xlabel('Recall')
    pyplot.ylabel('Precision')
    # show the legend
    pyplot.legend()
    # show the plot
    pyplot.show()
    

    # print(model.predict(trainX))
    # print(model.predict_log_proba(trainX))
    # print(model.predict_proba(trainX))

    pca = PCA(n_components=2).fit(trainX)
    pca_X = np.array(pca.transform(trainX))

    #probabilities = np.array(model.predict_proba(pca_X))
    positive_mask = np.array(trainy)
    negative_mask = []
    for label in trainy:
        if(label == 0):
            negative_mask += [1]
        else:
            negative_mask += [0]
    positive_mask = np.array([[a[0],a[0]] for a in positive_mask])
    negative_mask = np.array([[a,a] for a in negative_mask])
    positives = pca_X*positive_mask
    negatives = pca_X*negative_mask
    positives_x = [x[0] for x in positives]
    positives_y = [x[1] for x in positives]
    negatives_x = [x[0] for x in negatives]
    negatives_y = [x[1] for x in negatives]
    pyplot.scatter(negatives_x, negatives_y, marker='o', c ='b', s=2)
    pyplot.scatter(positives_x, positives_y, marker='x', c ='r', s=7)

    
    pyplot.show()
    print("end")





    # #NOTE:rest is compied from becnhmark program
    # result_dir = os.path.join(args.subjects_root_path, 'results')
    # try:
    #     os.makedirs(result_dir)
    # except:
    #     pass

    # file_name = 'result_scuffed_model'

    # with open(os.path.join(result_dir, 'train_{}.json'.format(file_name)), 'w') as res_file:
    #         ret = print_metrics_binary(data_Y, model.predict_proba(data_X))
    #         ret = {k : float(v) for k, v in ret.items()}
    #         json.dump(ret, res_file)


    # prediction = model.predict_proba(data_X_test)[:, 1]

    # with open(os.path.join(result_dir, 'test_{}.json'.format(file_name)), 'w') as res_file:
    #     ret = print_metrics_binary(data_Y_test, prediction)
    #     ret = {k: float(v) for k, v in ret.items()}
    #     json.dump(ret, res_file)

    #save_results(test_names, prediction, test_y, os.path.join(args.output_dir, 'predictions', file_name + '.csv'))



main()

