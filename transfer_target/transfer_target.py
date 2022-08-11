from simpletransformers.classification import (
    ClassificationModel, ClassificationArgs
)
from pycm import *
import pandas as pd
import numpy as np
import glob
import logging
import logger
import os, sys

logging = logging.getLogger(__name__)



def get_predictions(inputs, labels, cp, opt):
    model = ClassificationModel( "bert", cp )
    predictions, raw_outputs = model.predict(inputs)
    pred_file = cp + "transfer_predictions_target_" + opt + ".txt"

    with open(pred_file, "w") as out:
        for p in range(len(predictions)):
            out.write(str(predictions[p]) + "\t" + str(labels[p]) + "\t" + inputs[p] + "\n")
    return pred_file 



"""
Takes two lists with labels and reports prec, rec and f1.
"""
def p_r_f1(g, p, cp, opt):
    total = len(g); correct = 0
    dic = {}

    with open(cp + "transfer_results_" + opt + ".txt", "w") as out:

        for i in range(len(g)):
            if g[i] not in dic:
                dic[g[i]] = {'tp': 0, 'fp': 0, 'fn': 0}

            if p[i] not in dic:
                dic[p[i]] = {'tp': 0, 'fp': 0, 'fn': 0}
            # true positives
            if g[i] == p[i]:
                dic[g[i]]['tp'] += 1
                correct += 1
            # false negatives
            else:
                dic[g[i]]['fn'] += 1
                dic[p[i]]['fp'] += 1

        for l in dic:
            p = 0; r = 0; f1 = 0
            if (dic[l]['tp'] + dic[l]['fp']) > 0:
                p = dic[l]['tp'] / (dic[l]['tp'] + dic[l]['fp'])
            if (dic[l]['tp'] + dic[l]['fn']) > 0:
                r = dic[l]['tp'] / (dic[l]['tp'] + dic[l]['fn'])
            if (p + r) > 0:
                f1 = (2 * p * r) / (p + r)
            out.write("CLASS: " + str(l) + " " + str(round(p, 2)) + " " + str(round(r, 2)) + " " + str(round(f1, 2)) + "\n")
        out.write("ACC:\t" + str(correct/total) + "(" + str(correct) + "/" + str(total) + ")")

    return


def eval_cp(df, cp, opt):
    gold = []; pred = []
    for idx, row in df.iterrows():
        gold.append(row.gold)
        pred.append(row.pred)

    cm = ConfusionMatrix(gold, pred, digit=5)
    logging.info("MACRO:\tprec: %fs\trec: %fs\tf1: %fs", cm.PPV_Macro, cm.TPR_Macro, cm.F1_Macro)
    logging.info("MICRO:\tprec: %fs\trec: %fs\tf1: %fs", cm.PPV_Micro, cm.TPR_Micro, cm.F1_Micro)

    p_r_f1(gold, pred, cp, opt)
    return



def get_sentence_pairs(df):
    sents = []; tags = []
    for idx, row in df.iterrows():
        sents.append(row.text)
        tags.append(row.labels)
    return sents, tags



def get_sentences(df):
    sents = []; tags = []
    for idx, row in df.iterrows():
        sents.append(row.text)
        tags.append(row.labels)
    return sents, tags



########################################################################################
# specify model parameters
model_args = {
            "num_train_epochs": 50,
            "learning_rate": 1e-4,
            "manual_seed": 43, # seeds used for fold 1: 11, 2: 43, 3: 14, 4: 55, 5: 7
            "max_seq_length": 128,
            "use_early_stopping": True,
            "early_stopping_delta": 0.01,
            "early_stopping_metric": "eval_loss",
            "early_stopping_metric_minimize": True,
            "early_stopping_patience": 5,
            "evaluate_during_training": True,
            #"evaluate_during_training_verbose": True,
            "evaluate_during_training_steps": 1000,
            "use_cached_eval_features": False,
             }


folds = ['fold1', 'fold2', 'fold3', 'fold4', 'fold5'] 
opt = 'target'

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
    
    # For target prediction, we only train on titles and subtitles. If you want to include
    # the text body of the press releases, remove the # in the next 3 lines
    train_df['text'] = train_df['title'] + " " + train_df['subtitle'] #+ " " + train_df['body']
    dev_df['text']   = dev_df['title'] + " " + dev_df['subtitle'] #+ " " + dev_df['body']
    test_df['text']  = test_df['title'] + " " + test_df['subtitle'] #+ " " + test_df['body']

    train_df = train_df[['text', 'labels']]; train_df = train_df.astype({'labels': int})
    dev_df   = dev_df[['text', 'labels']];   dev_df   = dev_df.astype({'labels': int})
    test_df  = test_df[['text', 'labels']];  test_df  = test_df.astype({'labels': int})

    

    # Create a ClassificationModel
    model = ClassificationModel("bert", "bert-base-german-cased", num_labels=5, args=model_args) 


    # Train the model
    logging.info("Start training ...")
    model.train_model(train_df, eval_df=dev_df)
    logging.info("Training done...")

    dev_inputs, dev_tags = get_sentences(dev_df)
    test_inputs, test_tags = get_sentences(test_df) 
    
    # Make predictions with the best trained model
    checkpoints = []
    # uncomment the next line (and remove the line after)
    # if you want to evaluate all checkpoints:
    #checkpoints = glob.glob("./outputs/checkpoint-*/")
    
    checkpoints.append("./outputs/best_model/")

    for cp in checkpoints:
        pred_file_dev = get_predictions(dev_inputs, dev_tags, cp, "dev")
        res_dev_df = pd.read_csv(pred_file_dev, sep='\t', names=['pred', 'gold', 'text'])
        eval_cp(res_dev_df, cp, "dev")
        pred_file_test = get_predictions(test_inputs, test_tags, cp, "test")
        res_test_df = pd.read_csv(pred_file_test, sep='\t', names=['pred', 'gold', 'text'])
        eval_cp(res_test_df, cp, "test")

    # rename output folder
    os.rename('outputs', 'outputs.bert-base-german-cased.' + opt + '.' + fold)

