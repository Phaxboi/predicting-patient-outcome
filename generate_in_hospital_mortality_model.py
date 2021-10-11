#this file contains the methods used to run the regression model to predict in hopsital mortality

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
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
from matplotlib import pyplot
from prepare_features_mortality_pred import *



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--subjects_root_path', type=str, help='Directory containing subject subdirectories.')
    parser.add_argument('-use_generated_features_file', action='store_true', help='Used to specify if the previously generated features in root/result should be used.')
    parser.add_argument('-categorical', action='store_true', help='Set this if you want to run the categorical model instead of the numerical.')
    parser.add_argument('-use_word_embeddings', action='store_true', help='Set this to train models with word embeddings, preferably with \'use_generated_files\'')
    args = parser.parse_args()

    use_categorical_flag = args.categorical
    use_word_embeddings = args.use_word_embeddings

    #read already generated files
    if args.use_generated_features_file:
        if use_categorical_flag:
            features = pd.read_csv(os.path.join(args.subjects_root_path, 'result\\features_categorical.csv'))
            outcomes = pd.read_csv(os.path.join(args.subjects_root_path, 'result\\outcomes_categorical.csv'))
        else:
            features = pd.read_csv(os.path.join(args.subjects_root_path, 'result\\features_numerical.csv'))
            outcomes = pd.read_csv(os.path.join(args.subjects_root_path, 'result\\outcomes_numerical.csv'))

        data_X = features.values
        data_Y = outcomes.values
    
    else:
        #extract features for all patients
        (data_X, data_Y, ids) = extract_features(args.subjects_root_path, use_categorical_flag, use_word_embeddings)

        if(use_categorical_flag):
            features_file_name = 'features_categorical.csv'
            outcomes_file_name = 'outcomes_categorical.csv'
        else:
            features_file_name = 'features_numerical.csv'
            outcomes_file_name = 'outcomes_numerical.csv'

        #save X and Y data to 'result' map
        features = pd.DataFrame(data_X)
        features.to_csv(os.path.join(args.subjects_root_path, 'result\\', features_file_name), index=False)

        outcomes = pd.DataFrame(data_Y)
        outcomes.to_csv(os.path.join(args.subjects_root_path, 'result\\', outcomes_file_name), index=False)


        #id list, to ensure order is correct between standard features, outcomes and wv features
        ids_list = open(args.subjects_root_path + "/result/feature_outcomes_ids.csv", "w", encoding="utf16")
        for item in ids:
            ids_list.writelines(str(item) + "\n")
        ids_list.close()

    #read the raw data, concat new feature then re-scale the data
    if(use_word_embeddings):
        if(use_categorical_flag):
            features_raw = pd.read_csv(os.path.join(args.subjects_root_path, 'result\\features_categorical_raw.csv'))
        else:
            features_raw = pd.read_csv(os.path.join(args.subjects_root_path, 'result\\features_numerical_raw.csv'))

        wv_df = (pd.read_csv(args.subjects_root_path + '/result/wv_feature_file.csv'))
        data_X_raw = (pd.concat([features_raw, wv_df], axis=1)).values

        scaler = StandardScaler()
        scaler.fit(data_X_raw)
        data_X = scaler.transform(data_X_raw)


    data_Y = data_Y.ravel()
    trainX, testX, trainy, testy = train_test_split(data_X, data_Y, test_size=0.01, stratify=data_Y)#, random_state=2
    testy = np.array(testy)
    trainy = np.array(trainy)
    C = 1
    max_iter = 1000
    solver ='lbfgs'
    scoring = "f1"
    model = LogisticRegression(penalty='l2', C = C, max_iter=max_iter, solver=solver)#, random_state=42
    #model = LogisticRegressionCV(penalty='l2', Cs = [C], max_iter=1000, cv=10)
    #model = RandomForestClassifier(n_estimators= 1000)#, random_state= 42
    #model.fit(data_X, data_Y)

    scores = cross_val_score(model, trainX, trainy, cv = 10, scoring=scoring, n_jobs=-1)
    print(scores)
    print("Average "+ scoring +  " scoring: %0.3f (+/- %0.3f)" % (scores.mean(), scores.std()))

    print('C = %f' % C)

    #print('Test accuracy:'+ str(model.score(testX, testy)))

    

    # ns_probs = [0 for _ in range(len(testy))]
    # lr_probs = model.predict_proba(testX)
    # lr_probs = lr_probs[:, 1]


    # #confusion matrix for training data
    # cm = confusion_matrix(trainy, model.predict(trainX))
    # print(cm)
    # train_prec = cm[1][1] / (cm[1][1] + cm[0][1])
    # print('Training precision: %f' %train_prec)
    # test_recall = cm[1][1] / (cm[1][1] + cm[1][0])
    # print('Training recall: %f' %test_recall)
    
    # cm = confusion_matrix(testy, model.predict(testX))
    # print(cm)
    # test_prec = cm[1][1] / (cm[1][1] + cm[0][1])
    # print('Test precision: %f' %test_prec)
    # test_recall = cm[1][1] / (cm[1][1] + cm[1][0])
    # print('Test recall: %f' %test_recall)
    # cm_display = ConfusionMatrixDisplay(cm).plot()
    # pyplot.show()

    # # calculate scores
    # ns_auc = roc_auc_score(testy, ns_probs)
    # lr_auc = roc_auc_score(testy, lr_probs)
    # # summarize scores
    # #print('No Skill: ROC AUC=%.3f' % (ns_auc))
    # print('Test-data: ROC AUC=%.3f' % (lr_auc))
    # # calculate roc curves
    # ns_fpr, ns_tpr, _ = roc_curve(testy, ns_probs)
    # lr_fpr, lr_tpr, _ = roc_curve(testy, lr_probs)
    # # plot the roc curve for the model
    # pyplot.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
    # pyplot.plot(lr_fpr, lr_tpr, marker='.', label='Logistic')
    # # axis labels
    # pyplot.xlabel('False Positive Rate')
    # pyplot.ylabel('True Positive Rate')
    # # show the legend
    # pyplot.legend()
    # # show the plot
    # pyplot.show()

    # # predict probabilities
    # lr_probs = model.predict_proba(testX)
    # # keep probabilities for the positive outcome only
    # lr_probs = lr_probs[:, 1]
    # # predict class values
    # yhat = model.predict(testX)
    # lr_precision, lr_recall, _ = precision_recall_curve(testy, lr_probs)
    # lr_f1, lr_auc = f1_score(testy, yhat), auc(lr_recall, lr_precision)
    # # summarize scores
    # print('Logistic: f1=%.3f auc=%.3f' % (lr_f1, lr_auc))
    # # plot the precision-recall curves
    # positive_len = len(testy[testy==1])
    # total_len = len(testy)
    # no_skill = positive_len / total_len 
    # pyplot.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
    # pyplot.plot(lr_recall, lr_precision, marker='.', label='Logistic')
    # # axis labels
    # pyplot.xlabel('Recall')
    # pyplot.ylabel('Precision')
    # # show the legend
    # pyplot.legend()
    # # show the plot
    # pyplot.show()
    


    # pca = PCA(n_components=2).fit(trainX)
    # pca_X = np.array(pca.transform(trainX))

    # #probabilities = np.array(model.predict_proba(pca_X))
    # positive_mask = np.array(trainy)
    # negative_mask = []
    # for label in trainy:
    #     if(label == 0):
    #         negative_mask += [1]
    #     else:
    #         negative_mask += [0]
    # positive_mask = np.array([[a[0],a[0]] for a in positive_mask])
    # negative_mask = np.array([[a,a] for a in negative_mask])
    # positives = pca_X*positive_mask
    # negatives = pca_X*negative_mask
    # positives_x = [x[0] for x in positives]
    # positives_y = [x[1] for x in positives]
    # negatives_x = [x[0] for x in negatives]
    # negatives_y = [x[1] for x in negatives]
    # pyplot.scatter(negatives_x, negatives_y, marker='o', c ='b', s=2)
    # pyplot.scatter(positives_x, positives_y, marker='x', c ='r', s=7)

    
    # pyplot.show()
    print("end")



def concat_wv_features(data_X, subjects_root_path):
    wvs = pd.read_csv(subjects_root_path + 'result/wv_feature_file.csv')









main()

