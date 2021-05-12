#this file contains the methods used to run the regression model to predict in hopsital mortality

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import auc
from sklearn.model_selection import train_test_split

import os
import numpy as np
import argparse
import pandas as pd
from tqdm import tqdm
from metrics import *
from matplotlib import pyplot
from prepare_features_mortality_pred import scale_values

import re
import json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--subjects_root_path', type=str, help='Directory containing subject subdirectories.')
    args = parser.parse_args()

    (data_X, data_Y) = scale_values(args.subjects_root_path)
    trainX, testX, trainy, testy = train_test_split(data_X, data_Y, test_size=0.5, random_state=2)
    testy = np.array(testy)
    trainy = np.array(trainy)
    logreg = LogisticRegression(penalty='l2', C=0.01)#, random_state=42
    logreg.fit(trainX, trainy)
    print(logreg.score(testX, testy))



    ns_probs = [0 for _ in range(len(testy))]
    lr_probs = logreg.predict_proba(testX)
    lr_probs = lr_probs[:, 1]

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
    lr_probs = logreg.predict_proba(testX)
    # keep probabilities for the positive outcome only
    lr_probs = lr_probs[:, 1]
    # predict class values
    yhat = logreg.predict(testX)
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




    # #NOTE:rest is compied from becnhmark program
    # result_dir = os.path.join(args.subjects_root_path, 'results')
    # try:
    #     os.makedirs(result_dir)
    # except:
    #     pass

    # file_name = 'result_scuffed_model'

    # with open(os.path.join(result_dir, 'train_{}.json'.format(file_name)), 'w') as res_file:
    #         ret = print_metrics_binary(data_Y, logreg.predict_proba(data_X))
    #         ret = {k : float(v) for k, v in ret.items()}
    #         json.dump(ret, res_file)


    # prediction = logreg.predict_proba(data_X_test)[:, 1]

    # with open(os.path.join(result_dir, 'test_{}.json'.format(file_name)), 'w') as res_file:
    #     ret = print_metrics_binary(data_Y_test, prediction)
    #     ret = {k: float(v) for k, v in ret.items()}
    #     json.dump(ret, res_file)

    #save_results(test_names, prediction, test_y, os.path.join(args.output_dir, 'predictions', file_name + '.csv'))



main()

