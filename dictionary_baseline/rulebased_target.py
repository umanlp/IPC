import pandas as pd
import glob
import os, sys

from stop_words import get_stop_words

from tabulate import tabulate
from pycm import *
import glob
import spacy
import csv
import re
import time
import json
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, precision_score, recall_score, f1_score

from nltk.tokenize import word_tokenize


STOPWORDS = []

"""
Do some basic preprocessing based on spacy
(tokenisation, stopword removal, lemmatisation)
"""
def preproc(text, nlp, config):
    doc = nlp(text)
    if config["lemmatise"] == True:
        clean_text = " ".join([token.lemma_ for token in doc if token.lemma_ not in STOPWORDS and not token.is_punct])
    else:
        clean_text = " ".join([token.text for token in doc if token.text not in STOPWORDS and not token.is_punct])
    return clean_text




"""
Write results (F1 macro, micro) to file
"""
def write_results(macro, micro, fold, opt):
        outfile =  "rule_based_target_f1_"+fold+"." + opt + ".txt"
        with open(outfile, "w") as out:
                out.write("F1 (macro):\t" + str(macro) +"\n")
                out.write("F1 (micro):\t" + str(micro) +"\n")
        return


"""
Take the labels predicted by the model and write to file
"""
def write_preds(y, preds, fold, opt):
        outfile = "rule_based_target_predictions_"+fold+"." + opt + ".txt"
        with open(outfile, "w") as out:
                for i in range(len(y)):
                        out.write(str(y[i])+"\t"+str(preds[i])+"\n")
        return




"""
Look for party names in text. If we find one party name, let's assume that this is the target.
If we find more than one party names, then let's use the first one (other options: randomly
select one party name).
Return target party name.
"""
def rule_based(X, y, fold, idx2label, opt):
    preds = []
    # dictionary: define variants of party names
    parties = {
            'SPÖ': 1, 
            'Sozialdemokratische Partei Österreichs': 1, 
            'Sozialdemokratischen Partei Österreichs': 1,
            'ÖVP': 2, 
            'Österreichische Volkspartei': 2, 
            'Österreichischen Volkspartei': 2,
            'Neue Volkspartei': 2,
            'FPÖ': 3, 
            'Freiheitliche Partei Österreichs': 3, 
            'Freiheitlichen Partei Österreichs': 3,
            'die Grünen': 4, 
            'Die Grünen': 4,
            'Grünen': 4,
            'die Grüne Alternative': 4,
            'Die Grüne Alternative': 4,
            'Grüne': 4,
            }

    # look for party names in X
    for idx, row in X.iterrows():
        found = 0
        for p in parties:
            if p in row.text:
                if found == 0:
                    preds.append(parties[p])
                    found = 1
        if found == 0:
            preds.append(0)

    write_preds(y, preds, fold, opt)
    f1_macro = f1_score(y, preds, average="macro")
    f1_micro = f1_score(y, preds, average="micro")

    write_results(f1_macro, f1_micro, fold, opt)
    print("Rule-based results for targets:\t", fold, "\tf1 (macro)", f1_macro, "\tf1 (micro)", f1_micro)

    return





### We do a five-fold cross-validation
folds = ['fold1', 'fold2', 'fold3', 'fold4', 'fold5'] 

### Define the task (target prediction)
opt = 'target'

### Load German spacy model for preprocessing
nlp = spacy.load('de_core_news_sm')

### Specify experimental setting
config = {
    'lemmatise': False,     # turn on/off lemmatisation
    'use_stopwords': True,  # turn on/off stopword removal (doesn't really make sense to remove stopwords as we are using a dictionary ;)
    'use_body_text': False, # also use the body of the press releases to search for dictionary entries?
        }

if config['use_stopwords'] == True:
    STOPWORDS = get_stop_words('de')

### Mapping from indices to party names
idx2label= {
        0: 'NONE',
        1: 'SPÖ',
        2: 'ÖVP',
        3: 'FPÖ',
        4: 'Greens',
        }



for fold in folds:
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

    # run the rule-based approach on the test data 
    # (we only need test as we do not do any model development or parameter tuning) 
    rule_based(test_df, y_test, fold, idx2label, 'test')

