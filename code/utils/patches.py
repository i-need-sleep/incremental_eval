import types
import warnings
import copy

import torch.nn.functional as F
import torch
from torch.utils.data import DataLoader
from avalanche.training.utils import copy_params_dict, zerolike_params_dict, ParamData
from avalanche.models.utils import avalanche_forward


def patch(strategy, model, scenario, strategy_type):
    # Define patches
    def patched_unpack_minibatch(self, model=model):
        """Check if the current mini-batch has 3 components."""
        mbatch = self.mbatch
        assert mbatch is not None
        assert len(mbatch) >= 3

        # mbatch should be [(x), (y), (task_id)]
        # Convert mbatch for the comet preprocessing pipeline
        new_mbatch = [[], [0 for _ in mbatch[-1]], mbatch[-1]]
        converted = []
        for i in range(len(mbatch[0])):
            converted.append({
                'src': mbatch[0][i],
                'ref': mbatch[1][i],
                'neg': mbatch[2][i],
                'pos': mbatch[3][i],
            })
        new_mbatch[0] = converted

        mbatch = new_mbatch
        mbatch[0] = model.prepare_sample(mbatch[0])
        self.mbatch = mbatch

    def patched_forward(self, batch):
        src_input_ids = batch['src_input_ids'].to(self.device)
        src_attention_mask = batch['src_attention_mask'].to(self.device)
        ref_input_ids = batch['ref_input_ids'].to(self.device)
        ref_attention_mask = batch['ref_attention_mask'].to(self.device)
        pos_input_ids = batch['pos_input_ids'].to(self.device)
        pos_attention_mask = batch['pos_attention_mask'].to(self.device)
        neg_input_ids = batch['neg_input_ids'].to(self.device)
        neg_attention_mask = batch['neg_attention_mask'].to(self.device)

        src_sentemb = self.get_sentence_embedding(src_input_ids, src_attention_mask)
        ref_sentemb = self.get_sentence_embedding(ref_input_ids, ref_attention_mask)
        pos_sentemb = self.get_sentence_embedding(pos_input_ids, pos_attention_mask)
        neg_sentemb = self.get_sentence_embedding(neg_input_ids, neg_attention_mask)

        loss = self.loss(src_sentemb, pos_sentemb, neg_sentemb) + self.loss(
            ref_sentemb, pos_sentemb, neg_sentemb
        )

        distance_src_pos = F.pairwise_distance(pos_sentemb, src_sentemb)
        distance_ref_pos = F.pairwise_distance(pos_sentemb, ref_sentemb)
        # Harmonic mean between anchors and the positive example
        distance_pos = (2 * distance_src_pos * distance_ref_pos) / (
            distance_src_pos + distance_ref_pos
        )

        # Harmonic mean between anchors and the negative example
        distance_src_neg = F.pairwise_distance(neg_sentemb, src_sentemb)
        distance_ref_neg = F.pairwise_distance(neg_sentemb, ref_sentemb)
        distance_neg = (2 * distance_src_neg * distance_ref_neg) / (
            distance_src_neg + distance_ref_neg
        )

        return {
            "loss": loss,
            "distance_pos": distance_pos,
            "distance_neg": distance_neg,
        }
    
    def patched_criterion(self):
        return self.mb_output['loss']
    
    def patched_ewc_compute_importances(
        self, model, criterion, optimizer, dataset, device, batch_size, num_workers=0
    ):
        """
        Compute EWC importance matrix for each parameter
        """

        model.eval()

        # Set RNN-like modules on GPU to training mode to avoid CUDA error
        if device == "cuda":
            for module in model.modules():
                if isinstance(module, torch.nn.RNNBase):
                    warnings.warn(
                        "RNN-like modules do not support "
                        "backward calls while in `eval` mode on CUDA "
                        "devices. Setting all `RNNBase` modules to "
                        "`train` mode. May produce inconsistent "
                        "output if such modules have `dropout` > 0."
                    )
                    module.train()

        # list of list
        importances = zerolike_params_dict(model)
        collate_fn = dataset.collate_fn if hasattr(dataset, "collate_fn") else None
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=collate_fn,
            num_workers=num_workers,
        )
        for i, mbatch in enumerate(dataloader):
            # get only input, target and task_id from the batch

            new_mbatch = [[], [0 for _ in mbatch[-1]], mbatch[-1]]
            converted = []
            for i in range(len(mbatch[0])):
                converted.append({
                    'src': mbatch[0][i],
                    'ref': mbatch[1][i],
                    'neg': mbatch[2][i],
                    'pos': mbatch[3][i],
                })
            new_mbatch[0] = converted
            mbatch = new_mbatch
            mbatch[0] = model.prepare_sample(mbatch[0])
            batch = mbatch

            x, y, task_labels = batch[0], batch[1], batch[-1]

            optimizer.zero_grad()
            out = avalanche_forward(model, x, task_labels)
            loss = out['loss']
            loss.backward()

            for (k1, p), (k2, imp) in zip(
                model.named_parameters(), importances.items()
            ):
                assert k1 == k2
                if p.grad is not None:
                    imp.data += p.grad.data.clone().pow(2)

        # average over mini batch length
        for _, imp in importances.items():
            imp.data /= float(len(dataloader))

        model.train()

        return importances
    
    original_collate_fn = copy.deepcopy(scenario.train_stream[0].dataset.collate_fn)

    def handle_error_collate_fn(x):
        try:
            return original_collate_fn(x)
        except:
            return [('“作为一名教练，我要告诉是时候开始另一场比赛了。”',) * len(x), ('"As a coach, I would tell you it\'s time to run another play."',) * len(x), ("'As a coach, I'm going to tell it's time to start another game.'",) * len(x), torch.tensor([0 for _ in x])]

    # Patch stuff
    strategy._unpack_minibatch = types.MethodType(patched_unpack_minibatch, strategy)
    model.forward = types.MethodType(patched_forward, model)
    strategy.criterion = types.MethodType(patched_criterion, strategy)

    if strategy_type == 'ewc':
        strategy.plugins[0].compute_importances = types.MethodType(patched_ewc_compute_importances, strategy.plugins[0])

    # Dirty fix
    for i, exp in enumerate(scenario.train_stream):
        exp.dataset.collate_fn = handle_error_collate_fn
    return strategy