import pandas as pd
import numpy as np
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
Do some basic preprocessing, based on spacy
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
def write_results(macro, micro, fold):
        outfile =  "rule_based_stance_f1_"+fold+".test.txt"
        with open(outfile, "w") as out:
                out.write("F1 (macro):\t" + str(macro) +"\n")
                out.write("F1 (micro):\t" + str(micro) +"\n")
        return


"""
Take the labels predicted by the model and write to file
"""
def write_preds(y, preds, fold):
        outfile = "rule_based_stance_predictions_"+fold+".test.txt"
        with open(outfile, "w") as out:
                for i in range(len(y)):
                        out.write(str(y[i])+"\t"+str(preds[i])+"\n")
        return



def read_senti_lexicon(lex_file):
    lex_dic = { 'word': [], 'senti': [] }
    lex_df = pd.read_csv(lex_file, sep="\t")
    lex_dic['word'] = [x for x in lex_df['lemma']]
    lex_dic['senti'] = [s for s in lex_df['sentiment']]

    return lex_dic



"""
Use sentiment lexicon (sentiMerge) for rule-based sentiment prediction. Compute a score
for each text, based on the proportion of sum(scores(pos),scores(neg)) / #pos+#neg.

Return stance label.
lemma   PoS     sentiment       weight

"""
def rule_based(X, y, fold, lex_dic):
    preds = []; scores = []; texts = []
    # look for lemmas in the text that have a lexicon entry
    for idx, row in X.iterrows():
        texts.append(row.text)
        text = sorted(set(row.text.split()))
        idx = [(t, lex_dic['senti'][lex_dic['word'].index(t.lower())]) for t in text if t.lower() in lex_dic['word']]
        if len(idx) == 0:
            preds.append(0)
            scores.append(500)
        else:
            # we sum up the positive and negative sentiment scores for the words in the lexicon
            # and normalize by the number of lexicon terms found in the text
            score = sum([x[1] for x in idx])/len(idx)
            scores.append(score)
            
            # Confidence intervall (CI) threshold, based on 1,000 bootstraps of sample mean: 
            # percentiles 2.5, 97.5 [-0.0281, -0.0126]
            # percentiles 5, 95     [-0.0282, -0.0128]

            if score <= -0.0282:
                preds.append(-1)
            elif score >= 0.0128:
                preds.append(1)
            else:
                preds.append(0)


    if len(y) != len(scores):
        print("ERROR:", len(y), len(scores))
        sys.exit()

    #for i in range(len(y)):
    #    print(fold, "\t", y[i], "\t", scores[i], "\t", preds[i], "\t", texts[i])

    write_preds(y, preds, fold)
    #eval_predictions(y, preds, False)
    f1_macro = f1_score(y, preds, average="macro")
    f1_micro = f1_score(y, preds, average="micro")

    # Write results to file
    write_results(f1_macro, f1_micro, fold)
    print("Rule-based results for stance:\t", "\tf1 (macro)", f1_macro, "\tf1 (micro)", f1_micro)

    return





def get_predictions(inputs, labels, cp, opt):
    model = ClassificationModel( "bert", cp )
    print("inputs:", type(inputs))
    predictions, raw_outputs = model.predict(inputs)

    with open(cp + "predictions_" + opt + ".txt", "w") as out:
        for p in range(len(predictions)):
            out.write(str(predictions[p]) + "\t" + str(labels[p]) + "\t" + inputs[p] + "\n")
    return


def eval_model(df, cp, opt):
    model = ClassificationModel( "bert", cp )
    result, model_outputs, wrong_predictions = model.eval_model(df)
    with open(cp + "results_" + opt + ".txt", "w") as out:
        out.write(str(result))

    return






folds = ['fold1', 'fold2', 'fold3', 'fold4', 'fold5'] 
opt = 'stance'
lex_name = '../lexicon/sentimerge_nospin_no-null.txt'
lex_dic = read_senti_lexicon(lex_name)

# Load spacy model for preprocessing
nlp = spacy.load('de_core_news_sm')

# Define experimental settings
config = {
    'lemmatise': True,      # Turn on/off lemmatisation
    'use_stopwords': True,  # Turn on/off the use of stopwords
    'use_body_text': True,  # Use the text body of the press releases
        }                   # (if set to False, then use only titles and subtitles are used)

if config['use_stopwords'] == True:
    STOPWORDS = get_stop_words('de')


for fold in folds:
    data_dir = '../data/' + opt + '/' + fold

    train_df = pd.read_csv(data_dir + '/train.csv', sep='\t', error_bad_lines=False)
    dev_df   = pd.read_csv(data_dir + '/dev.csv', sep='\t', error_bad_lines=False)
    test_df  = pd.read_csv(data_dir + '/test.csv', sep='\t', error_bad_lines=False)


    train_df = train_df.rename(columns={'text': 'body', 'stance': 'labels'}).dropna()
    train_df = train_df.sample(frac=1).reset_index(drop=True)
    dev_df   = dev_df.rename(columns={'text': 'body', 'stance': 'labels'}).dropna()
    dev_df   = dev_df.sample(frac=1).reset_index(drop=True)
    test_df  = test_df.rename(columns={'text': 'body', 'stance': 'labels'}).dropna()
    
    if config['use_body_text']:
        train_df['text'] = train_df['title'] + " " + train_df['subtitle'] + " " + train_df['body']
        dev_df['text']   = dev_df['title'] + " " + dev_df['subtitle'] + " " + dev_df['body']
        test_df['text']  = test_df['title'] + " " + test_df['subtitle'] + " " + test_df['body']
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

    # run the rule-based approach on the test data (we only need test really) 
    rule_based(test_df, y_test, fold, lex_dic)
    #rule_based(dev_df, y_dev, fold, lex_dic)
