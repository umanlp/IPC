import pandas as pd
import numpy as np
import glob
import logging
import logger
import os, sys

from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, GridSearchCV
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import KFold, StratifiedKFold

from stop_words import get_stop_words

from tabulate import tabulate
import glob
import spacy
import csv
import re
import time
import json
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, precision_score, recall_score, f1_score
from nltk.tokenize import word_tokenize

logging = logging.getLogger(__name__)


STOPWORDS = []

"""
Do some basic preprocessing with spacy
(tokenisation, stopword removal, lemmatisation)
"""
def preproc(text, nlp, config):
    doc = nlp(text)
    if config["lemmatise"] == True:
        clean_text = " ".join([token.lemma_.lower() for token in doc if token.lemma_ not in STOPWORDS and not token.is_punct])
    else:
        clean_text = " ".join([token.text.lower() for token in doc if token.text not in STOPWORDS and not token.is_punct])
    return clean_text


"""
Takes a list with text (as strings) and 
uses the countvectorizer and chi^2 for 
feature selection.
Returns the features and feature names.
"""
def get_bow_with_feat_selection(X_train, X_dev, X_test, y, N):

        vec = TfidfVectorizer()
        X_train_vec = vec.fit_transform(X_train)
        X_dev_vec  = vec.transform(X_dev)
        X_test_vec  = vec.transform(X_test)
        start = time.time()
        chi2_selector = SelectKBest(chi2, k=N)
        X_train_vec = chi2_selector.fit_transform(X_train_vec, y)
        X_dev_vec  = chi2_selector.transform(X_dev_vec)
        X_test_vec  = chi2_selector.transform(X_test_vec)
        feature_names = vec.get_feature_names()
        feature_names = [feature_names[i] for i in chi2_selector.get_support(indices=True)]

        end = time.time()
        logging.info('Train vectorizer and do feature selection: %0.2f' % (end - start))
        return X_train_vec, X_dev_vec, X_test_vec, feature_names


"""
Take the gold labels y and the system predictions
and print evaluation results (acc, prec/rec/f1) and
confusion matrix. Also print gold and predicted labels
"""
def eval_predictions(y, predictions, write_preds, fold):

        logging.info("ACC:  %0.2f", accuracy_score(y, predictions))
        logging.info("PREC: %0.2f", precision_score(y, predictions, average='macro'))
        logging.info("REC:  %0.2f", recall_score(y, predictions, average='macro'))
        logging.info("F1:   %0.2f", f1_score(y, predictions, average='macro'))

        if write_preds == True:
            with open("preds_"+fold+".txt", "w") as out:
                for i in range(len(y)):
                    out.write(str(y[i]) + " " + str(predictions[i]) + "\n")


"""
Write results (F1 macro, micro) to file
"""
def write_results(macro, micro, fold, clf_name):
        outfile =  clf_name+"_target_f1_"+fold+".txt"
        with open(outfile, "w") as out:
                out.write("F1 (macro):\t" + str(macro) +"\n")
                out.write("F1 (micro):\t" + str(micro) +"\n")
        return


"""
Take the labels predicted by the model and write to file
"""
def write_preds(y, preds, fold, clf_name):
        outfile = clf_name+"_target_predictions_"+fold+".txt"
        with open(outfile, "w") as out:
                for i in range(len(y)):
                        out.write(str(y[i])+"\t"+str(preds[i])+"\n")
        return




"""
Train model and evaluate either on dev or on test set (if eval_dev is set to True,
then we use the dev set; otherwise the test set is used for evaluation)
"""
def run_svc(X_train, X_dev, X_test, y_train, y_dev, y_test, fold, eval_dev):

        clf =  LinearSVC(random_state=42)
        clf.fit(X_train, y_train)

        if eval_dev == True:
            preds = clf.predict(X_dev)
            score = clf.score(X_dev, y_dev)
            write_preds(y_dev, preds, fold, 'SVC-dev')
            eval_predictions(y_dev, preds, False, fold)
            f1_macro = f1_score(y_dev, preds, average="macro")
            f1_micro = f1_score(y_dev, preds, average="micro")
            write_results(f1_macro, f1_micro, fold, 'SVC-test')
        else:
            preds = clf.predict(X_test)
            score = clf.score(X_test, y_test)

            write_preds(y_test, preds, fold, 'SVC-test')
            eval_predictions(y_test, preds, False, fold)
            f1_macro = f1_score(y_test, preds, average="macro")
            f1_micro = f1_score(y_test, preds, average="micro")
            write_results(f1_macro, f1_micro, fold, 'SVC-test')

        logging.info("Results for LinearSVC\t%s\tf1 (macro) %0.2f\tf1 (micro)  %0.2f",fold, f1_macro, f1_micro)

        return








############################################################################
### We do a five-fold cross-validation
folds = ['fold1', 'fold2', 'fold3', 'fold4', 'fold5'] 

### The task is target prediction
opt = 'target'

### Load the German spacy model for preprocessing
nlp = spacy.load('de_core_news_sm')

### Specify the experimental setup
config = {
    'lemmatise': False,     # Turn on/off lemmatisation
    'use_stopwords': True,  # Turn on/off stopword removal
    'num_feat': 10000,      # Number of features for Chi2 feature selection
    'use_body_text': True,  # Also use the text body of the press releases 
        }                   # for training


if config['use_stopwords'] == True:
    STOPWORDS = get_stop_words('de')


for fold in folds:

    logging.info("Start training on: %s", fold)
    data_dir = '../data/' + opt + '/' + fold

    train_df = pd.read_csv(data_dir + '/train.csv', sep='\t', error_bad_lines=False)
    dev_df   = pd.read_csv(data_dir + '/dev.csv', sep='\t', error_bad_lines=False)
    test_df  = pd.read_csv(data_dir + '/test.csv', sep='\t', error_bad_lines=False)


    train_df = train_df.rename(columns={'text': 'body', 'target': 'labels'}).dropna()
    train_df = train_df.sample(frac=1).reset_index(drop=True)
    dev_df   = dev_df.rename(columns={'text': 'body', 'target': 'labels'}).dropna()
    dev_df   = dev_df.sample(frac=1).reset_index(drop=True)
    test_df  = test_df.rename(columns={'text': 'body', 'target': 'labels'}).dropna()
    
    if config['use_body_text']:
        train_df['text'] = train_df['title'] + train_df['subtitle'] + train_df['body']
        dev_df['text']   = dev_df['title'] + dev_df['subtitle'] + dev_df['body']
        test_df['text']  = test_df['title'] + test_df['subtitle'] + test_df['body']
    else:
        train_df['text'] = train_df['title'] + train_df['subtitle'] 
        dev_df['text']   = dev_df['title'] + dev_df['subtitle'] 
        test_df['text']  = test_df['title'] + test_df['subtitle'] 

    train_df = train_df[['text', 'labels']]; train_df = train_df.astype({'labels': int})
    dev_df   = dev_df[['text', 'labels']];   dev_df   = dev_df.astype({'labels': int})
    test_df  = test_df[['text', 'labels']];  test_df  = test_df.astype({'labels': int})

    train_df['text'] = [preproc(row.text, nlp, config) for idx, row in train_df.iterrows()]    
    dev_df['text']   = [preproc(row.text, nlp, config) for idx, row in dev_df.iterrows()] 
    test_df['text']  = [preproc(row.text, nlp, config) for idx, row in test_df.iterrows()] 

    y_train = list(train_df['labels'])
    y_dev   = list(dev_df['labels'])
    y_test  = list(test_df['labels'])

    X_train_vec, X_dev_vec, X_test_vec, feature_names = get_bow_with_feat_selection(train_df['text'], dev_df['text'], test_df['text'], y_train, config['num_feat'])
    
    # Train the model and evaluate on this fold
    # Set last argument to True if you want to evaluate on the dev set (this setting gives results for the test set)
    run_svc(X_train_vec,  X_dev_vec, X_test_vec, y_train, y_dev, y_test, fold, False)



