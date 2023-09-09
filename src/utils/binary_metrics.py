import numpy as np
import torch
from torchmetrics import Metric
from typing import List
import sys


class BINARY_F1(Metric):
    '''Micro F1 Score'''
    def __init__(self,
                 ignore_index: int = None,
                 constraint_type: str = 'BX',
                 dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.ignore_index = ignore_index
        if constraint_type.upper() == 'BX':
            self.constraint_type = constraint_type
            self.ids = dict({0: 'X', 1: 'B'})
            self.label2id = {v: k for k, v in self.ids.items()}
        else:
            raise NotImplementedError

        self.add_state('tp', default=torch.tensor(0), dist_reduce_fx='sum')
        self.add_state('fp', default=torch.tensor(0), dist_reduce_fx='sum')
        self.add_state('tn', default=torch.tensor(0), dist_reduce_fx='sum')
        self.add_state('fn', default=torch.tensor(0), dist_reduce_fx='sum')

    def __repr__(self):
        rep = 'BINARY_F1(ignore_index=`{}`, constraint_type=`{}`)'.format(
            self.ignore_index, self.constraint_type.upper())
        return rep

    def __state__(self):
        return 'tp: {},\nfp: {},\ntn: {}, fn: {}'.format(
            self.tp, self.fp, self.tn, self.fn)

    def update(self, preds: torch.tensor, golds: torch.tensor):
        if len(preds.shape) == 2 and len(golds.shape) == 2:
            for _pred, _gold in zip(preds.tolist(), golds.tolist()):
                self._update_states(_pred, _gold)
        elif len(preds.shape) == 1 and len(golds.shape) == 1:
            self._update_states(preds.tolist(), golds.tolist())
        else:
            print('tensor dim larger than 2 is not supported')
            sys.exit()

    def _update_states(self, pred: List, gold: List):
        if self.ignore_index is not None:
            pred, gold = self._remove_ignore_index(pred, gold)

        pred = np.array(pred)
        gold = np.array(gold)

        pos_pred = np.argwhere(pred == self.label2id['B'])
        neg_pred = np.argwhere(pred == self.label2id['X'])

        pos_pred = pos_pred[pos_pred < gold.shape[0]]
        neg_pred = neg_pred[neg_pred < gold.shape[0]]

        self.tp += int(np.sum(gold[pos_pred] == self.label2id['B']))
        self.fp += int(np.sum(gold[pos_pred] == self.label2id['X']))
        self.tn += int(np.sum(gold[neg_pred] == self.label2id['X']))
        self.fn += int(np.sum(gold[neg_pred] == self.label2id['B']))

    def compute(self):
        p = self.tp / (self.tp + self.fp)
        r = self.tp / (self.tp + self.fn)
        if p == r == 0.0:
            f = 0.0
        else:
            f = 2 * p * r / (p + r)
        return f

    def _remove_ignore_index(self, pred, gold):
        _pred = []
        _gold = []
        for p, g in zip(pred, gold):
            if g != self.ignore_index:
                _pred.append(p)
                _gold.append(g)
        return _pred, _gold
