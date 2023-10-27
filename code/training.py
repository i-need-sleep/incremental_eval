import os
import argparse
import datetime

import torch
import numpy as np
from scipy.stats import pearsonr
import avalanche
import comet

import utils.data_utils as data_utils
import utils.patches as patches
import utils.globals as uglobals

def train(args):
    # Data
    scenario = data_utils.make_scenario(debug=args.debug, oracle=args.strategy == 'oracle')

    # Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = comet.models.RankingMetric(learning_rate=args.lr, nr_frozen_epochs=0, validation_data=['placeholder']) # We only need to use the forward and inference functions
    print(f'Model loaded at {device}')
    
    # Strategy
    if args.strategy == 'naive' or args.strategy == 'oracle':
        strategy = avalanche.training.Naive(
            model=model,
            optimizer=torch.optim.Adam(model.parameters(), lr=args.lr), # Use a plain optimizer for simplicity (no scheduler, different lrs, freezing enc, etc.)
            criterion=None, # We use the criterion in the model
            train_mb_size=args.batch_size,
            train_epochs=args.n_epoch,
            eval_mb_size=args.batch_size,
            device=device,
            evaluator=avalanche.training.plugins.EvaluationPlugin(
                avalanche.evaluation.metrics.loss_metrics(minibatch=False, epoch=True, experience=True, stream=True), # Only log the loss for training
                loggers=[avalanche.logging.InteractiveLogger()],
            )
        )
    elif args.strategy == 'ewc':
        strategy = avalanche.training.EWC(
            model=model,
            optimizer=torch.optim.Adam(model.parameters(), lr=args.lr), # Use a plain optimizer for simplicity (no scheduler, different lrs, freezing enc, etc.)
            criterion=None, # We use the criterion in the model
            train_mb_size=args.batch_size,
            train_epochs=args.n_epoch,
            eval_mb_size=args.batch_size,
            device=device,
            evaluator=avalanche.training.plugins.EvaluationPlugin(
                avalanche.evaluation.metrics.loss_metrics(minibatch=False, epoch=True, experience=True, stream=True), # Only log the loss for training
                loggers=[avalanche.logging.InteractiveLogger()],
            ),
            ewc_lambda = 0.4,
        )
    elif args.strategy == 'replay':
        strategy = avalanche.training.Replay(
            model=model,
            optimizer=torch.optim.Adam(model.parameters(), lr=args.lr), # Use a plain optimizer for simplicity (no scheduler, different lrs, freezing enc, etc.)
            criterion=None, # We use the criterion in the model
            train_mb_size=args.batch_size,
            train_epochs=args.n_epoch,
            eval_mb_size=args.batch_size,
            device=device,
            evaluator=avalanche.training.plugins.EvaluationPlugin(
                avalanche.evaluation.metrics.loss_metrics(minibatch=False, epoch=True, experience=True, stream=True), # Only log the loss for training
                loggers=[avalanche.logging.InteractiveLogger()],
            ),
            mem_size=200
        )
    else:
        raise NotImplementedError
    
    # Patch the strategy
    patches.patch(strategy, model, strategy_type=args.strategy)

    # Train
    out = []
    for i, exp in enumerate(scenario.train_stream):
        model.train()
        strategy.train(exp)

        # Save the checkpoint
        save_dir = os.path.join(uglobals.CHECKPOINTS_DIR, args.name)
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f'exp_{i}.pt')
        torch.save({'model_state_dict': model.state_dict()}, save_path)
        
        # Eval
        eval(model, scenario.test_stream, args, i)
    return

def eval(model, test_stream, args, training_idx):
    model.eval()
    
    for exp_idx, exp in enumerate(test_stream):
        # Make a loader
        loader = torch.utils.data.DataLoader(exp.dataset, batch_size=args.batch_size, shuffle=False)
        with torch.no_grad():
            model_outs = []
            scores = []
            for i, batch in enumerate(loader): 
                # src, ref, pred, score, placeholder

                # Convert inputs
                converted = []
                for i in range(len(batch[0])):
                    converted.append({
                        'src': batch[0][i],
                        'ref': batch[1][i],
                        'neg': batch[2][i],
                        'pos': batch[2][i],
                    })
                model_in = model.prepare_sample(converted)

                # Forward
                model_out = model.forward(model_in)['distance_pos'].cpu().tolist()

                score = batch[3].cpu().tolist()
                model_outs += model_out
                scores += score
        
        # Get pearson correlation
        model_out = np.array(model_outs)
        model_out = np.nan_to_num(model_out)
        scores = np.array(scores)
        scores = np.nan_to_num(scores)

        pearson = pearsonr(model_outs, scores).statistic
        print(f'Training {training_idx}, Testing {exp_idx}, Pearson {pearson}')

        # Save the output
        save_dir = os.path.join(uglobals.OUTPUTS_DIR, args.name)
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f'train_{training_idx}_test_{exp_idx}_pearson_{round(pearson, 3)}.pt')
        torch.save({'model_outs': model_outs, 'scores': scores}, save_path)

    return


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--name', type=str, default='unnamed')
    parser.add_argument('--debug', action='store_true')

    # Formulation
    parser.add_argument('--strategy', default='', type=str) # naive, ewc, replay, oracle

    # Training
    parser.add_argument('--lr', default='3e-5', type=float)
    parser.add_argument('--batch_size', default='16', type=int)
    parser.add_argument('--n_epoch', default='1', type=int)

    args = parser.parse_args()

    # if args.debug:
        # args.name = 'debug'
        # args.n_epoch = 1

    print(args)
    train(args)