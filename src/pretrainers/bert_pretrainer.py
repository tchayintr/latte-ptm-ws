import abc
import argparse
from functools import partial
import os
from pathlib import Path
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import transformers
from typing import Any, List, Union

from core.modules.crf import CRF
from utils.data.datamodule import DataModule
from utils.data.dataset import (BertDataset, pad_sequence, ids2chunks,
                                ids2tokens)
from utils.word_oov_cws_metrics import CWS_WORD_F1, CWS_OOV_RECALL
from utils.binary_metrics import BINARY_F1

BERT_CHAR_INPUT_KEYS = ['input_ids', 'attention_mask', 'token_type_ids']
BERT_CHAR_INPUT_ID_KEYS = 'input_ids'
ORG_INPUT_KEY = 'org_ids'
CHAR_ORG_IDS = 'org_ids'


class BertPretrainer(pl.LightningModule):

    def __init__(self, hparams, *args, **kwargs):
        super(BertPretrainer, self).__init__()
        self.save_hyperparameters(hparams)

        if ((self.hparams.node_comp_type == 'wv'
             and self.hparams.run == 'latte')
                and not self.hparams.generate_unigram_node):
            raise AssertionError(
                '`--node-comp-type==wv` for `--run==latte` requires' +
                '`--generate-unigram-node`')

        self.data_module = None
        self._setup_data_module()

        self.criterion = None
        self._setup_criterion()

        self.metric = None
        self.oov_metric = None
        self._setup_metrics()

        # bert: config and model
        bert_config = transformers.AutoConfig.from_pretrained(
            self.hparams.pretrained_model, output_hidden_states=True)
        self.bert = transformers.AutoModel.from_pretrained(
            self.hparams.pretrained_model, config=bert_config)

        # bert: resize if any token is added
        self.resize_bert_embeddings(
            self.bert,
            vocab_size=self._get_tokenizer_vocab_size(),
            org_vocab_size=bert_config.vocab_size)

        # bert: freezing
        if self.hparams.freeze:
            print('[INFO]: freezing BERT weight')
            self.freeze_bert(self.bert)

        self.hparams.hidden_size = self.bert.config.hidden_size
        if self.hparams.bert_mode == 'concat':
            self.hparams.hidden_size *= 4

        # datasets
        self.train_set, self.valid_set = self._get_train_and_valid_data()

        # evaluation
        self.use_bin_eval = self.hparams.metric_type == 'word-bin'

        # ddp
        self.use_ddp = True if self.hparams.num_gpus > 1 else False

    @abc.abstractmethod
    def forward(self) -> Any:
        pass

    @abc.abstractmethod
    def training_step(self, batch, batch_idx=None):
        pass

    @abc.abstractmethod
    def valid_step(self, batch, batch_idx=None):
        pass

    @abc.abstractmethod
    def test_step(self, batch, batch_idx=None):
        pass

    @abc.abstractmethod
    def validation_epoch_end(self, outputs):
        pass

    @abc.abstractmethod
    def test_epoch_end(self, outputs):
        pass

    @abc.abstractmethod
    def configure_optimizers(self):
        pass

    def on_train_start(self):
        self.logger.log_hyperparams(self.hparams, {
            'hp/': 0,
        })

    def _setup_data_module(self):
        params = self._get_data_module_params()
        data_module = DataModule(**params)
        data_module.setup()
        self.data_module = data_module

    def _setup_criterion(self):
        params = self._get_criterion_params()
        if params['criterion_type'] == 'crf-nll':
            self.criterion = CRF(constraint_type='BMES',
                                 ignore_index_for_mask=0)
        else:
            raise NotImplementedError

    def _setup_metrics(self):
        params = self._get_metric_params()
        dist_sync_on_step = True if self.hparams.num_gpus > 1 else False
        if params['metric_type'] == 'word-cws':
            self.metric = CWS_WORD_F1(ignore_index=-1,
                                      constraint_type='BMES',
                                      dist_sync_on_step=dist_sync_on_step)
            self.oov_metric = CWS_OOV_RECALL(
                vocab=self.data_module._train_vocab,
                dist_sync_on_step=dist_sync_on_step)
        elif params['metric_type'] == 'word-bin':
            self.metric = CWS_WORD_F1(ignore_index=-1,
                                      constraint_type='BMES',
                                      dist_sync_on_step=dist_sync_on_step)
            self.bin_metric = BINARY_F1(ignore_index=-1,
                                        constraint_type='BX',
                                        dist_sync_on_step=dist_sync_on_step)
        else:
            raise NotImplementedError

        self.oov_metric = CWS_OOV_RECALL(vocab=self.data_module._train_vocab,
                                         dist_sync_on_step=dist_sync_on_step)

    def _get_vocab(self):
        return self.data_module.vocab

    def _get_tokenizer_vocab_size(self):
        return len(self.data_module.tokenizer)

    def _get_tokenizer_padding_idx(self):
        return self.data_module.tokenizer_padding_idx

    def _get_train_and_valid_data(self):
        return self.data_module.train_set, self.data_module.valid_set

    def _get_feats_from_bert_outputs(self, outputs):
        if self.hparams.bert_mode == 'none':
            feats = outputs[0]
        elif self.hparams.bert_mode == 'concat':
            feats = outputs[2][-4:]
            feats = torch.cat(feats, dim=-1)
        elif self.hparams.bert_mode == 'sum':
            feats = outputs[2][-4:]
            feats = torch.stack(feats, dim=0).sum(dim=0)
        elif self.hparams.bert_mode == 'sum-all':
            feats = outputs[2][:]
            feats = torch.stack(feats, dim=0).sum(dim=0)
        else:
            raise ValueError
        return feats

    def _get_char_input_lengths(self, inputs):
        '''get org_ids and convert it into tensor of input length'''
        org_ids = self.get_char_input_org_ids(inputs)
        char_input_lengths = torch.sum(org_ids != 0, dim=-1)
        return char_input_lengths

    def _get_org_ids(self, inputs):
        return inputs[ORG_INPUT_KEY]

    def _construct_binary_labels(self, ts, ys) -> torch.Tensor:
        '''
        bmes_dict = {'B': 0, 'M': 1, 'E': 2, 'S': 3}
        '''
        preds = ts.detach().clone()
        golds = ys.detach().clone()

        pred_b_mask = preds == 0
        pred_m_mask = preds == 1
        pred_e_mask = preds == 2
        pred_s_mask = preds == 3
        preds[pred_b_mask] = 1
        preds[pred_m_mask] = 0
        preds[pred_e_mask] = 0
        preds[pred_s_mask] = 1

        gold_b_mask = golds == 0
        gold_m_mask = golds == 1
        gold_e_mask = golds == 2
        gold_s_mask = golds == 3
        golds[gold_b_mask] = 1
        golds[gold_m_mask] = 0
        golds[gold_e_mask] = 0
        golds[gold_s_mask] = 1

        return preds, golds

    def _get_dataloader_params(self, train=True):
        include_lattice = self.hparams.run == 'latte'
        build_dynamic_graph = False
        dynamic_graph_dropout = 0.0
        if train:
            build_dynamic_graph = (not self.hparams.unuse_dynamic_graph
                                   and self.hparams.graph_dropout > 0.0)
            dynamic_graph_dropout = self.hparams.graph_dropout
        return dict({
            'batch_size':
            self.hparams.batch_size,
            'shuffle':
            False if self.use_ddp else train,
            'num_workers':
            int(os.cpu_count() / (4 if not self.use_ddp else 8)),
            # 1,
            'pin_memory':
            True,
            'collate_fn':
            partial(BertDataset.generate_batch,
                    vocab=self._get_vocab(),
                    include_lattice=include_lattice,
                    build_dynamic_graph=build_dynamic_graph,
                    dynamic_graph_dropout=dynamic_graph_dropout)
        })

    def _get_data_module_params(self):
        return dict({
            'data_dir': self.hparams.data_dir,
            'ext_dic_file': self.hparams.ext_dic_file,
            'wv_model_path': self.hparams.wv_model_path,
            'batch_size': self.hparams.batch_size,
            'model_max_seq_length': self.hparams.model_max_seq_length,
            'max_token_length': self.hparams.max_token_length,
            'lang': self.hparams.lang,
            'normalize_unicode': self.hparams.normalize_unicode,
            'seed': self.hparams.seed,
            'pretrained_model': self.hparams.pretrained_model,
            'pretrained_save_path': self.hparams.pretrained_save_path,
            'shuffle_data': self.hparams.shuffle_data,
            'generate_unigram_node': self.hparams.generate_unigram_node,
            'include_dataset_token': self.hparams.include_dataset_token,
            'include_unc_token': self.hparams.include_unc_token,
            'include_lattice': self.hparams.run == 'latte',
            'build_dynamic_graph': not self.hparams.unuse_dynamic_graph,
            'include_valid_vocab': not self.hparams.include_valid_vocab,
            'train_split_ratio': self.hparams.train_split_ratio,
            'unc_token_ratio': self.hparams.unc_token_ratio,
            'node_comp_type': self.hparams.node_comp_type,
            'graph_dropout': self.hparams.graph_dropout,
            'use_binary': self.hparams.use_binary,
        })

    def _get_criterion_params(self):
        return dict({'criterion_type': self.hparams.criterion_type})

    def _get_metric_params(self):
        return dict({'metric_type': self.hparams.metric_type})

    def train_dataloader(self):
        params = self._get_dataloader_params(train=True)
        sampler = DistributedSampler(self.train_set,
                                     shuffle=True) if self.use_ddp else None
        return DataLoader(self.train_set, sampler=sampler, **params)

    def val_dataloader(self):
        params = self._get_dataloader_params(train=False)
        sampler = DistributedSampler(self.valid_set,
                                     shuffle=False) if self.use_ddp else None
        return DataLoader(self.valid_set, sampler=sampler, **params)

    def test_dataloader(self):
        raise NotImplementedError

    def _compute_active_loss(self, xs, ys, ps):
        if self._get_criterion_params()['criterion_type'] == 'crf-nll':
            active_loss = xs['attention_mask'] == 1
            active_logits = ps
            ignore_label_index = torch.tensor(
                self.criterion.ignore_index_for_mask).type_as(ys)
            active_labels = torch.where(active_loss, ys, ignore_label_index)
            log_likelihood = self.criterion(active_logits, active_labels,
                                            active_loss)
            loss = -log_likelihood / ys.size(0)  # mean
        else:
            active_loss = xs['attention_mask'].view(-1) == 1
            active_logits = ps.view(-1, 4)
            ignore_label_index = torch.tensor(
                self.criterion.ignore_index).type_as(ys)
            active_labels = torch.where(active_loss, ys.view(-1),
                                        ignore_label_index)
            loss = self.criterion(active_logits, active_labels)
        return loss

    def _tagging(self, xs, ys, ps):
        if self._get_criterion_params()['criterion_type'] == 'crf-nll':
            mask = xs['attention_mask'] != 0
            outputs = self.criterion.viterbi_tags(ps, mask)
            return pad_sequence([
                torch.tensor(tag, device=ps.device) for tag, score in outputs
            ])
        else:
            return torch.argmax(ps, dim=2)

    def _pad_ignore_tokens(self, ys, ts, ignore_label_index=-1):
        pad_labels = ys == ignore_label_index
        ts[pad_labels] = ignore_label_index
        return ts

    def _pad_special_label_tokens(self, ts, ys, org_ids) -> torch.Tensor:
        offset = 1
        if self.hparams.include_dataset_token:
            offset = 2
        seq_mask = torch.sum(org_ids != 0, dim=-1)
        ts[:, :offset] = -1
        ys[:, :offset] = -1
        for batch_idx, seq_len in enumerate(seq_mask):
            ts[batch_idx, seq_len + offset] = -1
            ys[batch_idx, seq_len + offset] = -1
        return ts, ys

    def _reconstruct_org_seqs(self, ts, ys,
                              org_ids) -> Union[List[str], List[str]]:
        len_offset = 1
        if self.hparams.include_dataset_token:
            len_offset = 2
        vocab = self._get_vocab()
        seqs = [ids2tokens(ids.tolist(), vocab) for ids in org_ids]
        preds, golds = [], []
        for seq, pred_ids, gold_ids in zip(seqs, ts.tolist(), ys.tolist()):
            '''
            seq: len=L
            seq + special_tokens: len=L+offset
            loc(seq w/o padding): [offset:L+offset]
            special_tokens: [CLS], [SEP], [DATASET]
            '''
            pred_ids = pred_ids[len_offset:len(seq) + len_offset]
            gold_ids = gold_ids[len_offset:len(seq) + len_offset]
            str_pred = ' '.join(ids2chunks(pred_ids, seq))
            str_gold = ' '.join(ids2chunks(gold_ids, seq))
            preds.append(str_pred)
            golds.append(str_gold)
        return preds, golds

    @rank_zero_only
    def save_pretrained_bert(self, path):
        self.bert.save_pretrained(path)

    @staticmethod
    def get_char_input_org_ids(inputs):
        return inputs[CHAR_ORG_IDS]

    @staticmethod
    def resize_bert_embeddings(pretrained_model, vocab_size: int,
                               org_vocab_size: int):
        if vocab_size != org_vocab_size:
            pretrained_model.resize_token_embeddings(vocab_size)

    @staticmethod
    def get_bert_char_inputs(inputs):
        return dict({
            data: inputs[data]
            for data in inputs if data in BERT_CHAR_INPUT_KEYS
        })

    @staticmethod
    def get_bert_char_input_ids(inputs):
        return inputs[BERT_CHAR_INPUT_ID_KEYS]

    @staticmethod
    def add_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser],
                                         add_help=False)
        parser.add_argument('--data-dir', type=Path)
        parser.add_argument('--ext-dic-file', type=Path)
        parser.add_argument('--save-dir', type=Path)
        parser.add_argument('--model-name')
        parser.add_argument('--model-version', type=int)
        parser.add_argument('--batch-size', type=int, default=32)
        parser.add_argument('--max-epochs', type=int, default=5)
        parser.add_argument('--accumulate-grad-batches', type=int, default=1)
        parser.add_argument('--gradient-clip-val', type=float, default=5.0)
        parser.add_argument('--model-max-seq-length', type=int, default=512)
        parser.add_argument('--lr', type=float, default=2e-5)
        parser.add_argument('--dropout', type=float, default=0.2)
        parser.add_argument('--pretrained-model',
                            choices=[
                                'data/ptm/bert-base-chinese',
                                'data/ptm/bert-base-japanese-char-v2',
                                'data/ptm/bert-base-multilingual-cased',
                                'bert-base-chinese',
                                'cl-tohoku/bert-base-japanese-char-v2',
                                'bert-base-multilingual-cased',
                            ])
        parser.add_argument('--bert-mode',
                            choices=['none', 'concat', 'sum', 'sum-all'],
                            default='none')
        parser.add_argument('--freeze', action='store_true')
        parser.add_argument('--bert-lr', type=float, default=2e-5)
        parser.add_argument('--optimized-decay',
                            action='store_true',
                            help='weight decay on subset of model params')
        parser.add_argument('--scheduler',
                            action='store_true',
                            help='use scheduler to control lr value')
        parser.add_argument('--lr-decay-rate', type=float, default=0.99)
        parser.add_argument('--shuffle-data', action='store_true')
        parser.add_argument('--include-dataset-token', action='store_true')
        parser.add_argument('--include-unc-token', action='store_true')
        parser.add_argument('--include-valid-vocab', action='store_true')
        parser.add_argument('--train-split-ratio', type=float, default=0.9)
        parser.add_argument('--unc-token-ratio', type=float, default=0.1)
        parser.add_argument(
            '--pretrained-save-path',
            type=Path,
            required=True,
            help='Specify a path to save pretrained model (tokenizer/bert)')

        return parser
