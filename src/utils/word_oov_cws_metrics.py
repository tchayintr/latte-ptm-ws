import torch
from torchmetrics import Metric
from typing import Dict, List, Tuple
import sys
'''
https://arxiv.org/abs/1906.12035
https://github.com/acphile/MCCWS/blob/master/utils.py
'''


class CWS_WORD_F1(Metric):
    '''Micro F1 Score'''
    def __init__(self,
                 ignore_index: int = None,
                 constraint_type: str = 'BMES',
                 dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.ignore_index = ignore_index
        if constraint_type.upper() == 'BMES':
            self.constraint_type = constraint_type
            self.ids = dict({0: 'B', 1: 'M', 2: 'E', 3: 'S'})
        else:
            raise NotImplementedError

        # (# w_ref ∩ w_hyp)
        self.add_state('correct_preds',
                       default=torch.tensor(0),
                       dist_reduce_fx='sum')
        # (# w_hyp)
        self.add_state('total_preds',
                       default=torch.tensor(0),
                       dist_reduce_fx='sum')
        # (# w_ref)
        self.add_state('total_correct',
                       default=torch.tensor(0),
                       dist_reduce_fx='sum')

    def __repr__(self):
        rep = 'WORD_CWS_F1(ignore_index=`{}`, constraint_type=`{}`)'.format(
            self.ignore_index, self.constraint_type.upper())
        return rep

    def __state__(self):
        return 'correct_preds: {},\ntotal_preds: {},\n'.format(
            self.correct_preds, self.total_preds) + 'total_correct: {}'.format(
                self.total_correct)

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
        pred = self._convert_ids_to_labels(pred)
        gold = self._convert_ids_to_labels(gold)

        pred_spans = set(self._convert_labels_to_spans(pred))
        gold_spans = set(self._convert_labels_to_spans(gold))

        self.correct_preds += len(gold_spans & pred_spans)
        self.total_preds += len(pred_spans)
        self.total_correct += len(gold_spans)

    def compute(self):
        p = (self.correct_preds /
             self.total_preds) if self.correct_preds > 0 else 0
        r = (self.correct_preds /
             self.total_correct) if self.correct_preds > 0 else 0
        f = 2 * p * r / (p + r) if p + r > 0 else 0
        return f

    def _remove_ignore_index(self, pred, gold):
        _pred = []
        _gold = []
        for p, g in zip(pred, gold):
            if g != self.ignore_index:
                _pred.append(p)
                _gold.append(g)
        return _pred, _gold

    def _convert_ids_to_labels(self, ids: List[int]) -> List[str]:
        return [self.ids[id] for id in ids]

    def _convert_labels_to_spans(self, labels: List[str]) -> List[Tuple]:
        spans = []
        if len(labels) == 0:
            return spans
        span = (0, 0)

        for i, label in enumerate(labels):
            if i == 0:
                span = (0, 0)
            elif label.upper() == 'B' or label.upper() == 'S':
                spans.append(span)
                span = (i, 0)
            span = (span[0], span[1] + 1)
        if span[1] != 0:
            spans.append(span)
        return spans


class CWS_OOV_RECALL(Metric):
    '''OOV recall'''
    def __init__(self, vocab: Dict, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.vocab = vocab

        self.add_state('recall', default=torch.tensor(0), dist_reduce_fx='sum')
        self.add_state('tot', default=torch.tensor(0), dist_reduce_fx='sum')

    def __repr__(self):
        rep = 'OOV_CWS_RECALL(vocab=`{}`)'.format(len(self.vocab))
        return rep

    def __state__(self):
        return 'recall: {},\ntot: {},\n'.format(self.recall, self.tot)

    def update(self, preds: List[str], golds: List[str]):
        for _pred, _gold in zip(preds, golds):
            self._update_states(_pred.split(), _gold.split())

    def _update_states(self, pred: List[str], gold: List[str]):
        i, j, id = 0, 0, 0
        for g in gold:
            if g not in self.vocab:
                self.tot += 1
                while i + len(pred[id]) <= j:
                    i += len(pred[id])
                    id += 1
                if i == j and len(
                        pred[id]) == len(g) and g.find(pred[id]) != -1:
                    self.recall += 1
            j += len(g)

    def compute(self):
        oov_recall = 1.0 * self.recall / self.tot
        return oov_recall
