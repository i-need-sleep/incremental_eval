import os
import random

import pandas as pd
import torch
import avalanche

import utils.globals as uglobals

def preprocess_data(dev_ratio=0.1, test_ratio=0.1):
    # Organize everything into (src, pred, ref, score) and make splits
    # First: Convertion into (src, pred, ref, score)
    year_dfs = {} # {year: df}
    
    year_dfs[2022] = convert_mqm_22(f'{uglobals.RAW_DIR}/mqm/mqm_generalMT2022_zhen.avg_seg_scores.tsv')

    for year in [2021, 2020]:
        year_dfs[year] = convert_mqm_21(f'{uglobals.RAW_DIR}/mqm/wmt-zhen-newstest{year}.csv')

    for year in range(2019, 2016, -1):
        year_dfs[year] = convert_da(f'{uglobals.RAW_DIR}/da/{year}-da.csv')

    
    # Second: Make splits
    for year in range(2017, 2023):
        df = year_dfs[year]

        # Split by src
        srcs = list(df['src'].unique())
        random.shuffle(srcs)

        num_srcs = len(srcs)
        num_dev = int(dev_ratio * num_srcs)
        num_test = int(test_ratio * num_srcs)
        
        srcs_dev = srcs[:num_dev]
        srcs_test = srcs[num_dev:num_dev+num_test]
        srcs_train = srcs[num_dev+num_test:]

        df_dev = df[df['src'].isin(srcs_dev)]
        df_test = df[df['src'].isin(srcs_test)]
        df_train = df[df['src'].isin(srcs_train)]

        df_dev.to_csv(f'{uglobals.PROCESSED_DIR}/{year}_dev.csv', index=False)
        df_test.to_csv(f'{uglobals.PROCESSED_DIR}/{year}_test.csv', index=False)
        df_train.to_csv(f'{uglobals.PROCESSED_DIR}/{year}_train.csv', index=False)
    return

def convert_da(path):
    df = pd.read_csv(path)
    # zhen only
    df = df[df['lp'] == 'zh-en']

    dict_out = {
        'src': [],
        'pred': [],
        'ref': [],
        'score': []
    }
    
    for i in range(len(df)):
        line = df.iloc[i]
        dict_out['src'].append(line['src'])
        dict_out['pred'].append(line['mt'])
        dict_out['ref'].append(line['ref'])
        dict_out['score'].append(line['score'])

    df_out = pd.DataFrame(dict_out)
    return df_out

def convert_mqm_21(path):
    df = pd.read_csv(path)

    dict_out = {
        'src': [],
        'pred': [],
        'ref': [],
        'score': []
    }
    
    for i in range(len(df)):
        line = df.iloc[i]
        dict_out['src'].append(line['src'])
        dict_out['pred'].append(line['mt'])
        dict_out['ref'].append(line['ref'])
        dict_out['score'].append(line['score'])

    df_out = pd.DataFrame(dict_out)
    return df_out

def convert_mqm_22(path):
    df = pd.read_table(path, on_bad_lines='skip')
    
    dict_out = {
        'src': [],
        'pred': [],
        'ref': [],
        'score': []
    }
    
    for i in range(len(df)):
        line = df.iloc[i]
        dict_out['src'].append(line['source'])
        dict_out['pred'].append(line['hyp'])
        dict_out['ref'].append(line['ref'])
        dict_out['score'].append(line['score'])

    df_out = pd.DataFrame(dict_out)
    return df_out

class RelativeRankDataset(torch.utils.data.Dataset):
    def __init__(self, path, debug=False, type='train'):
        self.debug = debug
        self.df = pd.read_csv(path)

        if type == 'train':
            self.data = self.make_pairs(self.df)
        elif type == 'test':
            self.data = self.make_test(self.df)
        else:
            raise ValueError(f'Unknown type {type}')

        print(f'Loaded {len(self.data)} samples from {path}, debug={self.debug}, type={type}')

        year = path[len(uglobals.PROCESSED_DIR)+1: len(uglobals.PROCESSED_DIR)+5]
        self.targets = [year] * len(self.data) # Dummy for Avalanche

    def make_pairs(self, df):
        pairs = [] # [{src, ref, pos, neg}]

        srcs = list(df['src'].unique())
        random.shuffle(srcs)

        for src in srcs:
            src_df = df[df['src'] == src]
            for i in range(len(src_df)):
                for j in range(i+1, len(src_df)):
                    line_i = src_df.iloc[i]
                    line_j = src_df.iloc[j]
                    
                    # Skip if the scores are the same
                    if line_i['score'] == line_j['score']:
                        continue

                    if line_i['score'] < line_j['score']:
                        worse_pred = line_i['pred']
                        better_pred = line_j['pred']
                    
                    else:
                        worse_pred = line_j['pred']
                        better_pred = line_i['pred']

                    pairs.append([src, line_i['ref'], worse_pred, better_pred])
            if self.debug:
                pairs = pairs[:2000]
                # break
        random.shuffle(pairs)
        return pairs
    
    def make_test(self, df):
        out = [] # [[src, ref, pred, score]]
        for i in range(len(df)):
            line = df.iloc[i]
            out.append([line['src'], line['ref'], line['pred'], line['score']])

            if self.debug and i >500:
                break
        return out
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, i):
        return self.data[i]
    
def make_scenario(debug=False, oracle=False):
    train_datasets = [RelativeRankDataset(f'{uglobals.PROCESSED_DIR}/{year}_train.csv', debug=debug) for year in range(2017, 2023)]
    dev_datasets = [RelativeRankDataset(f'{uglobals.PROCESSED_DIR}/{year}_dev.csv', debug=debug, type='test') for year in range(2017, 2023)]

    if oracle:
        # Pool all train/dev data together
        for datasets in [train_datasets, dev_datasets]:
            for dataset in datasets[1:]:
                datasets[0].data += dataset.data
                datasets[0].targets += dataset.targets
            random.shuffle(datasets[0].data)
        train_datasets = train_datasets[:1]
        dev_datasets = dev_datasets[:1]

    scenario = avalanche.benchmarks.generators.dataset_benchmark(train_datasets, dev_datasets)
    return scenario