import os
import argparse
import datetime

import pandas.core.indexes.numeric
import torch
import numpy as np
from scipy.stats import pearsonr
import comet
import avalanche
from avalanche.training.determinism.rng_manager import RNGManager
from avalanche.training.checkpoint import maybe_load_checkpoint, save_checkpoint

import utils.data_utils as data_utils
import utils.patches as patches
import utils.misc_utils as misc_utils
import utils.globals as uglobals

def train(args):

    # Seeding
    RNGManager.set_random_seeds(21)

    # Data
    scenario = data_utils.make_scenario(debug=args.debug, oracle=args.strategy == 'oracle')
    if args.anchor in ['worse', 'score_diff']:
        eval_on_train_scenario = data_utils.make_eval_on_train_scenario(debug=args.debug, oracle=args.strategy == 'oracle')

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
            ewc_lambda = args.ewc_lambda,
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
            mem_size=args.replay_size,
        )
    elif args.strategy == 'lwf':
        strategy = avalanche.training.LwF(
            model=model,
            optimizer=torch.optim.Adam(model.parameters(), lr=args.lr), # Use a plain optimizer for simplicity (no scheduler, different lrs, freezing enc, etc.)
            criterion=None, # We use the criterion in the model
            alpha=args.lwf_alpha,
            temperature=1,
            train_mb_size=args.batch_size,
            train_epochs=args.n_epoch,
            eval_mb_size=args.batch_size,
            device=device,
            evaluator=avalanche.training.plugins.EvaluationPlugin(
                avalanche.evaluation.metrics.loss_metrics(minibatch=False, epoch=True, experience=True, stream=True), # Only log the loss for training
                loggers=[avalanche.logging.InteractiveLogger()],
            ),
        )
    else:
        raise NotImplementedError
    
    # Load checkpoints
    # Strategy
    exp_counter = -1
    if args.strategy_checkpoint != '':
        
        # Re-patch the strategy
        patches.patch(args, strategy, model, scenario, strategy_type=args.strategy, anchor=args.anchor)

        strategy, exp_counter = maybe_load_checkpoint(strategy, args.strategy_checkpoint, map_location=device)
        print(f'Loading from checkpoint {args.strategy_checkpoint}, exp_counter: {exp_counter}')

    # Patch the strategy
    patches.patch(args, strategy, model, scenario, strategy_type=args.strategy, anchor=args.anchor)

    # Make/clean up the checkpoint and output directories
    checkpoint_dir = os.path.join(uglobals.CHECKPOINTS_DIR, args.name)
    misc_utils.make_and_clean_up_dirs(checkpoint_dir)
    output_dir = os.path.join(uglobals.OUTPUTS_DIR, args.name)
    misc_utils.make_and_clean_up_dirs(output_dir)

    # Train
    for i, exp in enumerate(scenario.train_stream):
        if i < exp_counter:
            continue

        if args.anchor in ['worse', 'score_diff'] and (i > 0 or args.debug) :
            # Get the predcitions on the training set
            anchor_preds = eval(model, eval_on_train_scenario.train_stream, args, i, save=False, eval_exp_idx=i)
            model.anchor_preds = anchor_preds
            model.anchor = args.anchor
            print(f'Anchor Len {len(anchor_preds)}')

        print(f'Training {i}')
        model.train()
        strategy.train(exp)

        # Save the checkpoint
        save_path = os.path.join(checkpoint_dir, f'exp_{i}.pt')
        torch.save({'model_state_dict': model.state_dict()}, save_path)

        # Save the strategy
        strategy_save_path = os.path.join(checkpoint_dir, f'exp_{i}_strat.pt')
        save_checkpoint(strategy, strategy_save_path, exclude=['model'])
        print(f'Strategy saved at: {strategy_save_path}')
        
        # Eval
        model.anchor = '' # Do not use anchors for eval
        eval(model, scenario.test_stream, args, i)
    return

def eval(model, test_stream, args, training_idx, save=True, eval_exp_idx=None):
    model.eval()
    
    for exp_idx, exp in enumerate(test_stream):
        if eval_exp_idx != None and exp_idx != eval_exp_idx:
            continue

        # Make a loader
        loader = torch.utils.data.DataLoader(exp.dataset, batch_size=args.batch_size, shuffle=False)
        with torch.no_grad():
            model_outs = []
            scores = []
            out_keys = [] # input_ids
            out = {} # {input_ids: emb}
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
                model_out = model.forward(model_in)
                distance = model_out['distance_pos'].cpu().tolist()
                emb = model_out['embedding_neg'].cpu()

                score = batch[3].cpu().tolist()
                model_outs += distance
                scores += score

                for idx in range(emb.shape[0]):
                    key = model_in['neg_input_ids'][idx]
                    # Remove 1 paddings
                    key = str(key[key != 1])
                    out[key] = emb[idx]
                    out_keys.append(key)

        # Get pearson correlation
        pearson = pearsonr(model_outs, scores).statistic * -1 # COMET-RANK is a distance metric, so we need to invert the correlation
        print(f'Training {training_idx}, Testing {exp_idx}, Pearson {pearson}')

        # Save the output
        if save:
            save_dir = os.path.join(uglobals.OUTPUTS_DIR, args.name)
            save_path = os.path.join(save_dir, f'train_{training_idx}_test_{exp_idx}_pearson_{round(pearson, 3)}.pt')
            torch.save({'model_outs': model_outs, 'scores': scores}, save_path)

        if args.anchor == 'score_diff':
            # Normalize the model and human scores
            model_outs = np.array(model_outs)
            scores = np.array(scores)
            model_outs = (model_outs - np.mean(model_outs)) / np.std(model_outs)
            scores = (scores - np.mean(scores)) / np.std(scores)

            # Get the absolute score differences
            score_diff = abs(scores - model_outs)

            # Find the indices of the smallest score_diffs
            top_score_diffs = np.argsort(score_diff)[: args.score_diff_anchor_size]

            # Keep only the items associated with the smalleset score_diffs
            out_keys = np.array(out_keys)
            out_keys = out_keys[top_score_diffs]
            out_keys = out_keys.tolist()
            
            out_out = {}
            for key in out_keys:
                out_out[key] = out[key]
            out = out_out

    return out


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--name', type=str, default='unnamed')
    parser.add_argument('--debug', action='store_true')

    # Strategy
    parser.add_argument('--strategy', default='', type=str) # naive, ewc, replay, oracle, lwf

    parser.add_argument('--anchor', default='', type=str) # worse, score_diff
    parser.add_argument('--anchor_loss_scale', default='1', type=float)
    parser.add_argument('--score_diff_anchor_size', default='3000', type=int)

    parser.add_argument('--ewc_lambda', default='0.4', type=float)
    parser.add_argument('--replay_size', default='1000', type=int)
    parser.add_argument('--lwf_alpha', default='1', type=float)

    # Training
    parser.add_argument('--lr', default='3e-6', type=float)
    parser.add_argument('--batch_size', default='16', type=int)
    parser.add_argument('--n_epoch', default='3', type=int)

    # Checkpoints
    parser.add_argument('--strategy_checkpoint', type=str, default='')

    args = parser.parse_args()

    if args.debug:
        args.name = 'debug'
        args.strategy = 'lwf'
        args.anchor = ''
        args.n_epoch = 1
        args.score_diff_anchor_size = 5
        # args.strategy_checkpoint = '../results/checkpoints/replay/exp_4_strat.pt'
    print(args)
    train(args)