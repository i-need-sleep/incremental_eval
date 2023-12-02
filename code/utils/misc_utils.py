import os
import torch
from scipy.stats import pearsonr
import pandas as pd

def parse_performance(s):
    out = []
    for line in s.split('\n'):
        score = line.split('Pearson ')[-1]
        out.append(float(score))

    print(out)
    print(f'Average score: {round(sum(out)/len(out), 3)}')

def make_and_clean_up_dirs(dir):
    os.makedirs(dir, exist_ok=True)
    for f in os.listdir(dir):
        os.remove(os.path.join(dir, f))

def pooled_eval(dir, prefix='train_5'):
    model_outs = []
    scores = []
    for file in os.listdir(dir):
        if prefix in file and ('test_3' in file or 'test_4' in file or 'test_5' in file):
            loaded = torch.load(f'{dir}/{file}')
            model_outs += loaded['model_outs']
            scores += loaded['scores']

    pearson = pearsonr(model_outs, scores).statistic * -1 # COMET-RANK is a distance metric, so we need to invert the correlation
    print(f'Pearson: {round(pearson, 3)}')

def convert_csv_to_txts(path, out_root):
    df = pd.read_csv(path)
    srcs = df['src'].to_list()
    refs = df['ref'].to_list()
    preds = df['pred'].to_list()

    # Make sure everything is a string
    srcs = [str(src) for src in srcs]
    refs = [str(ref) for ref in refs]
    preds = [str(pred) for pred in preds]

    # Write to linebreak-separated txt
    with open(f'{out_root}_src.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(srcs))
    with open(f'{out_root}_ref.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(refs))
    with open(f'{out_root}_pred.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(preds))
    return

def get_pearsons_corr_from_files(pred_path, data_path):
    true_ratings = pd.read_csv(data_path)[:500]['score'].to_list()
    
    # Convert txt data from pred_path to a list
    with open(pred_path, 'r', encoding='utf-8') as f:
        preds = f.read().split('\n')

    # Convert preds to floats
    preds = [float(pred) for pred in preds]

    # Compute Pearson's correlation
    pearson = pearsonr(preds, true_ratings).statistic
    print(pearson)
    # return pearson
    return preds, true_ratings