import pandas as pd
from pycm import *
import glob, sys

""" Write output to file """
def write_to_file(df, outfile):
    idx2label= {
        0: 'NONE',
        1: 'SPÖ',
        2: 'ÖVP',
        3: 'FPÖ',
        4: 'Greens',
        }

    with open(outfile, "w") as out:
        for idx, row in df.iterrows():
            out.write(idx2label[row.gold] + "\t" + idx2label[row.pred] + "\t" + row.text + "\n")
    return



"""
Takes two lists with labels and reports prec, rec and f1.
"""
def p_r_f1(g, p):
    total = len(g); correct = 0
    dic = {}
    for i in range(len(g)):
        if g[i] not in dic:
            dic[g[i]] = {'tp': 0, 'fp': 0, 'fn': 0}

        if p[i] not in dic:
            dic[p[i]] = {'tp': 0, 'fp': 0, 'fn': 0}
        # tp
        if g[i] == p[i]:
            dic[g[i]]['tp'] += 1
            correct += 1
        # fn
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
        print("CLASS:", l, str(round(p, 2)), str(round(r, 2)), str(round(f1, 2)))
    print("ACC:\t", correct/total, "(", correct, "/", total, ")")


"""
Takes an array of data frames and returns a new data frame
with the most frequently predicted label (ensemble vote).
"""
def get_ensemble_vote(preds):
    df = preds[0]
    df['e1'] = df['pred']
    df['e2'] = preds[1]['pred']
    df['e3'] = preds[2]['pred']
    df['e4'] = preds[3]['pred']
    df['e5'] = preds[4]['pred']
    
    for idx, row in df.iterrows():
        labels = [row.e1, row.e2, row.e3, row.e4, row.e5]
        df.at[idx,'pred']= max(set(labels), key = labels.count)
    return df


"""
Takes a df and evaluatespredictions:
pred_cat    gold_cat    ...
"""
def eval_cp(df):
    gold = []; pred = []
    for idx, row in df.iterrows():
        gold.append(row.gold)
        pred.append(row.pred)

    cm = ConfusionMatrix(gold, pred, digit=5)
    print("MACRO:\t", cm.PPV_Macro, "\t", cm.TPR_Macro, "\t", cm.F1_Macro)
    print("MICRO:\t", cm.PPV_Micro, "\t", cm.TPR_Micro, "\t", cm.F1_Micro)

    p_r_f1(gold, pred)
    return


###################################################################
opt   = 'test' # either train or test
seeds = ['run1_11', 'run2_43', 'run3_14', 'run4_55', 'run5_07']

for f in range(1, 6):
    preds = []
    for s in seeds:
        infile = s + '/outputs.bert-base-german-cased.target.fold' + str(f) + '/best_model/transfer_predictions_target_' + opt + '.txt'
        outfile = infile.split('/')[-1]
        outfile = outfile.replace('.txt', '_ensemble.' + str(f) + '.txt')

        preds.append(pd.read_csv(infile, sep='\t', names=['pred', 'gold', 'text']))

    df = get_ensemble_vote(preds)
    write_to_file(df, outfile)    
    eval_cp(df)


