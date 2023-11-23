import os
import torch
from scipy.stats import pearsonr

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