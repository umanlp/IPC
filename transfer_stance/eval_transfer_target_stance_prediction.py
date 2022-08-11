import pandas as pd
from pycm import *
import glob, sys
import logging
import logger

logging = logging.getLogger(__name__)



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
    #print((2 * cm.TPR_Macro * cm.PPV_Macro)/(cm.PPV_Macro + cm.TPR_Macro), cm.F1_Macro)
    print("MICRO:\t", cm.PPV_Micro, "\t", cm.TPR_Micro, "\t", cm.F1_Micro)
    #print((2 * cm.TPR_Micro * cm.PPV_Micro)/(cm.PPV_Micro + cm.TPR_Micro), cm.F1_Micro)

    p_r_f1(gold, pred)
    return
    #cm.print_matrix()


"""
Takes two df with target and stance labels (gold and pred)
and combines them into one atomic label.
Returns a df with the merged labels.
"""
def merge_ensemble_labels(df1, df2):
    df2['gold'] = df2['gold'].replace({2: '_neg'})
    df2['gold'] = df2['gold'].replace({0: '_neut'})
    df2['gold'] = df2['gold'].replace({1: '_pos'})

    df2['pred'] = df2['pred'].replace({2: '_neg'})
    df2['pred'] = df2['pred'].replace({0: '_neut'})
    df2['pred'] = df2['pred'].replace({1: '_pos'})

    df1['gold'] = df1['gold'].astype(str) + df2['gold'].astype(str)
    df1['pred'] = df1['pred'].astype(str) + df2['pred'].astype(str)

    # fix NONE labels (we only want NONE, no stance for NONE)
    df1['gold'] = df1['gold'].replace({'NONE_none': 'NONE',
        'NONE_neut': 'NONE',
        'NONE_pos':  'NONE',
        'NONE_neg':  'NONE'})
    df1['pred'] = df1['pred'].replace({'NONE_none': 'NONE',
        'NONE_neut': 'NONE',
        'NONE_pos':  'NONE',
        'NONE_neg':  'NONE'})

    return df1


"""
Takes two df with target and stance labels (gold and pred)
and combines them into one atomic label.
Returns a df with the merged labels.
"""
def merge_labels(df1, df2, opt):
    idx2label= {
        0: 'NONE',
        1: 'SPÖ',
        2: 'ÖVP',
        3: 'FPÖ',
        4: 'Greens',
        }
    df1['gold'] = df1['gold'].replace({0: 'NONE'})
    df1['gold'] = df1['gold'].replace({1: 'SPÖ'})
    df1['gold'] = df1['gold'].replace({2: 'ÖVP'})
    df1['gold'] = df1['gold'].replace({3: 'FPÖ'})
    df1['gold'] = df1['gold'].replace({4: 'Greens'})

    df1['pred'] = df1['pred'].replace({0: 'NONE'})
    df1['pred'] = df1['pred'].replace({1: 'SPÖ'})
    df1['pred'] = df1['pred'].replace({2: 'ÖVP'})
    df1['pred'] = df1['pred'].replace({3: 'FPÖ'})
    df1['pred'] = df1['pred'].replace({4: 'Greens'})

    if opt == 'bert':

        df2['gold'] = df2['gold'].replace({2: '_neg'})
        df2['gold'] = df2['gold'].replace({3: '_none'})
        df2['gold'] = df2['gold'].replace({0: '_neut'})
        df2['gold'] = df2['gold'].replace({1: '_pos'})
 
        df2['pred'] = df2['pred'].replace({2: '_neg'})
        df2['pred'] = df2['pred'].replace({3: '_none'})
        df2['pred'] = df2['pred'].replace({0: '_neut'})
        df2['pred'] = df2['pred'].replace({1: '_pos'})
    else:
        df2['gold'] = df2['gold'].replace({-1: '_neg'})
        df2['gold'] = df2['gold'].replace({-2: '_none'})
        df2['gold'] = df2['gold'].replace({0: '_neut'})
        df2['gold'] = df2['gold'].replace({1: '_pos'})


        df2['pred'] = df2['pred'].replace({-1: '_neg'})
        #df2['pred'] = df2['pred'].replace({-2: '_none'})
        df2['pred'] = df2['pred'].replace({0: '_neut'})
        df2['pred'] = df2['pred'].replace({1: '_pos'})


    df1['gold'] = df1['gold'].astype(str) + df2['gold'].astype(str)
    df1['pred'] = df1['pred'].astype(str) + df2['pred'].astype(str)

    # fix NONE labels (we only want NONE, no stance for NONE)
    df1['gold'] = df1['gold'].replace({'NONE_none': 'NONE', 
        'NONE_neut': 'NONE',
        'NONE_pos':  'NONE',
        'NONE_neg':  'NONE'})
    df1['pred'] = df1['pred'].replace({'NONE_none': 'NONE',
        'NONE_neut': 'NONE',
        'NONE_pos':  'NONE',
        'NONE_neg':  'NONE'})

    return df1


""" Write output to file """
def write_to_file(df, outfile):
    with open(outfile, "w") as out:
        for idx, row in df.iterrows():
            out.write(row.gold + "\t" + row.pred + "\n")
    return



#############################################################################

target_infile = sys.argv[1]
stance_infile = sys.argv[2]
ensemble = sys.argv[3]

outfile = target_infile.split('/')[-1]
if ensemble == True:
    outfile = outfile.replace('target_', 'target_stance_ensemble_')
else:
    outfile = outfile.replace('target_', 'target_stance_')

if ensemble == True:
    logging.info("\tReading ensemble predictions: %s", target_infile)
    target_df = pd.read_csv(target_infile, sep='\t', names=['gold', 'pred', 'text'])
    stance_df = pd.read_csv(stance_infile, sep='\t', names=['gold', 'pred', 'text'])

    logging.info("\tMerging labels...")
    df = merge_ensemble_labels(target_df, stance_df, 'bert')

else:
    logging.info("\tReading predictions: %s", target_infile)
    target_df = pd.read_csv(target_infile, sep='\t', names=['pred', 'gold', 'text'])
    stance_df = pd.read_csv(stance_infile, sep='\t', names=['pred', 'gold', 'text'])

    logging.info("\tMerging labels...")
    df = merge_labels(target_df, stance_df, 'bert')

logging.info("\tEvaluating target and stance predictions...")
eval_cp(df)
logging.info("\tWriting output to file: %s", outfile)
write_to_file(df, outfile)


