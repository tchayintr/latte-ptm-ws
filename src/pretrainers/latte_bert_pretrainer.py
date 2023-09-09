import argparse
from pathlib import Path
import torch
import torch.nn as nn
from typing import List

from core.modules.attention import Attention
from core.modules.gnn import LatticeEncoder
from pretrainers.bert_pretrainer import BertPretrainer
from utils.data.dataset import pad_sequence, lattice2cspans

LATTICE_INPUT_KEY = 'lattice'
MAPPING_TABLE_KEY = 'mapping'


class LatteBertPretrainer(BertPretrainer):

    def __init__(self, hparams, *args, **kwargs):
        super(LatteBertPretrainer, self).__init__(hparams)
        # wv
        wv_node_embed_size = 0
        if self.hparams.node_comp_type == 'wv':
            '''fasttext 300'''
            wv_node_embed_size = 300
        self.hparams.hidden_size += wv_node_embed_size

        # gnn
        node_embed_size = self.hparams.hidden_size
        self.node_embed = self.bert
        self.gnn = LatticeEncoder(
            embed_size=node_embed_size,
            hidden_size=self.hparams.hidden_size,
            num_layers=self.hparams.num_layers,
            num_heads=self.hparams.gnn_num_heads,
            dropout=self.hparams.dropout,
            gnn_type=self.hparams.gnn_type,
            add_self_loops=True,
            bidirectional=not self.hparams.unidirectional_gnn,
            shallow=self.hparams.shallow_gnn)

        # attention
        attn_feat_size = 1
        self.attention = Attention(node_embed_size,
                                   self.hparams.hidden_size,
                                   attn_comp_type=self.hparams.attn_comp_type,
                                   inner_dropout=self.hparams.attn_dropout)
        if self.hparams.attn_comp_type == 'wavg':
            attn_feat_size = 2
        elif self.hparams.attn_comp_type == 'wcon':
            attn_feat_size = sum(
                [i for i in range(1, self.hparams.max_token_length + 1)]) + 1

        # classifier
        classifier_in = (self.hparams.hidden_size * attn_feat_size)
        self.classifier = nn.Linear(classifier_in, 4)

        # dropout
        self.dropout = nn.Dropout(self.hparams.dropout)

    def forward(self, inputs, *args, **kwargs):
        # unigrams
        bert_char_inputs = self.get_bert_char_inputs(inputs)
        char_outputs = self.bert(**bert_char_inputs)
        outputs = self._get_feats_from_bert_outputs(char_outputs)
        if self.training:
            outputs = self.dropout(outputs)

        # tokens (gnn)
        lattice_inputs = self._get_lattice_inputs(inputs)
        lattice_outputs = self._embed_lattice_tokens(lattice_inputs)
        '''whether to use contextualised char from char_outputs'''
        if not self.hparams.unuse_context_char_node:
            '''
            replace context-free char in graph with
            contextualised char from char_outputs
            '''
            char_input_lengths = self._get_char_input_lengths(inputs)
            lattice_outputs = self._replace_char_node_attrs(
                lattice_outputs, outputs, char_input_lengths)
        '''whether to concat nodes with wv'''
        if self.hparams.node_comp_type == 'wv':
            outputs = self._concat_outputs_with_wv(outputs, lattice_outputs)
            lattice_outputs = self._concat_node_attrs_with_wv(lattice_outputs)
        lattice_outputs = self.gnn(lattice_outputs)

        # attention
        node_attrs = self._get_node_attrs_from_lattice(lattice_outputs)
        mapping_inputs = self._get_mapping_inputs(inputs)
        mapping_table, mapping_mask = self._get_mapping_table_and_mask(
            node_attrs, mapping_inputs, padding_idx=0)
        outputs, _ = self.attention(outputs, mapping_table, mapping_mask)

        # transformation
        outputs = self.classifier(outputs)
        return outputs

    def training_step(self, batch, batch_idx=None):
        xs, ys = batch
        ps = self.forward(xs)
        loss = self._compute_active_loss(xs, ys, ps)
        self.log('train/loss',
                 loss,
                 on_epoch=True,
                 prog_bar=False,
                 logger=True,
                 sync_dist=True,
                 rank_zero_only=True)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx=None):
        xs, ys = batch
        ps = self.forward(xs)
        loss = self._compute_active_loss(xs, ys, ps)
        ts = self._tagging(xs, ys, ps)
        ts = self._pad_ignore_tokens(ys, ts)

        org_ids = self._get_org_ids(xs)
        ts, ys = self._pad_special_label_tokens(ts, ys, org_ids)
        ts_str, ys_str = self._reconstruct_org_seqs(ts, ys, org_ids)

        self.metric(ts, ys)
        self.oov_metric(ts_str, ys_str)

        if self.use_bin_eval:
            ts_bin, ys_bin = self._construct_binary_labels(ts, ys)
            self.bin_metric(ts_bin, ys_bin)

        self.log('valid_loss',
                 loss,
                 on_epoch=True,
                 prog_bar=True,
                 logger=True,
                 sync_dist=True,
                 rank_zero_only=True)
        return {'valid_loss': loss}

    def validation_end(self, outputs):
        avg_loss = torch.stack([output['valid_loss']
                                for output in outputs]).mean()
        return {'valid_loss': avg_loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([output['valid_loss']
                                for output in outputs]).mean()
        self.log('valid_loss',
                 avg_loss,
                 on_epoch=True,
                 prog_bar=True,
                 logger=True,
                 sync_dist=True,
                 rank_zero_only=True)
        self.log('valid_f1',
                 self.metric.compute(),
                 on_epoch=True,
                 prog_bar=True,
                 logger=True,
                 sync_dist=True,
                 rank_zero_only=True)
        self.log('valid_oov_recall',
                 self.oov_metric.compute(),
                 on_epoch=True,
                 prog_bar=True,
                 logger=True,
                 sync_dist=True,
                 rank_zero_only=True)
        self.metric.reset()
        self.oov_metric.reset()
        if self.use_bin_eval:
            self.log('valid_bin_F1',
                     self.bin_metric.compute(),
                     on_epoch=True,
                     prog_bar=True,
                     logger=True,
                     sync_dist=True,
                     rank_zero_only=True)
            self.bin_metric.reset()
        return

    def _embed_lattice_tokens(self, inputs):
        lattice = inputs.detach().clone()
        token_ids = lattice.token_id
        lattice.x = self._get_token_embeddings(token_ids)
        return lattice

    def _get_token_embeddings(self, token_ids: torch.tensor):
        outputs = self._get_feats_from_bert_outputs(self.node_embed(token_ids))
        return torch.squeeze(outputs)

    def _get_node_attrs_from_lattice(self, batch) -> torch.Tensor:
        return pad_sequence(
            [data.x.detach().clone() for data in batch.to_data_list()])

    def _get_mapping_table_and_mask(self, node_attrs, mapping, padding_idx=0):
        '''
        node_attrs: node embeddings from graph
        mapping: indexes for character-to-candidate tokens
        return:
            mapping_table: table to keep for each batch which has n characters,
                and for each character has candidate tokens with
                the embed_size dimension.
                e.g. mapping_table.shape = (2, 5, 10, 300) means
                    there are 2 batches, each batch has at most 5 characters,
                    each character has at most 10 candidate tokens with 300
                    feature dimension
            mask: a tensor to ignore padding token in mapping
        '''
        batch_size, n_tokens, embed_size = node_attrs.shape
        batch_size, n_unigrams, n_cand_tokens = mapping.shape
        mask = mapping != padding_idx

        flatten_mapping = mapping.view(batch_size, -1)
        maps = [
            node_attrs[batch_idx][indices]
            for batch_idx, indices in enumerate(flatten_mapping)
        ]

        mapping_table = torch.stack(maps)
        mapping_table = mapping_table.view(batch_size, n_unigrams,
                                           n_cand_tokens, embed_size)
        return mapping_table, mask

    def _get_lattice_inputs(self, inputs):
        return inputs[LATTICE_INPUT_KEY]

    def _get_mapping_inputs(self, inputs):
        return inputs[MAPPING_TABLE_KEY]

    def _get_wv_model(self):
        return self.data_module.wv_model

    def _get_tokenizer(self):
        return self.data_module.tokenizer

    def _get_char_node_ids_from_lattice(self, batch_lattice) -> List:
        '''unigrams need to be placed sequencially'''
        char_node_ids = []
        for i, data in enumerate(batch_lattice.to_data_list()):
            '''[BOS], [EOS], and dataset_token'''
            special_node_ids = [
                node_idx
                for node_idx, (start_idx, end_idx) in enumerate(data.span)
                if (end_idx -
                    start_idx == 0 and (start_idx >= 0 and end_idx >= 0))
            ]
            '''unigrams'''
            _char_node_ids = [
                node_idx
                for node_idx, (start_idx, end_idx) in enumerate(data.span)
                if end_idx - start_idx == 1
            ]
            if self.hparams.include_dataset_token:
                _char_node_ids = [special_node_ids[2]] + _char_node_ids
            _char_node_ids = [special_node_ids[0]
                              ] + _char_node_ids + [special_node_ids[1]]
            char_node_ids.append(_char_node_ids)
        return char_node_ids

    def _get_char_node_attrs_from_lattice(self, batch_lattice) -> torch.Tensor:
        char_node_ids = self._get_char_node_ids_from_lattice(batch_lattice)
        char_node_attrs = []
        for node_ids, data in zip(char_node_ids, batch_lattice.to_data_list()):
            node_attrs = data.x.detach().clone()
            char_node_attrs.append(node_attrs[node_ids])
        return pad_sequence([attrs for attrs in char_node_attrs])

    def _get_char_node_indices(self, batch_lattice, batch, input_lengths):
        '''get all char node indices for referring to base chars'''
        batch_lattice_cspans = [
            lattice2cspans(lattice)
            for lattice in batch_lattice.to_data_list()
        ]
        char_node_indices = []
        '''offset to shift spans for special_tokens'''
        offset = (2 if self.hparams.include_dataset_token else 1)
        for cspans, input_len in zip(batch_lattice_cspans,
                                     input_lengths.tolist()):
            indices = []
            for cspan in cspans:
                char_node_idx = cspan[0]
                char_node_idx += offset
                indices.append(char_node_idx)
            '''
            add special_tokens indices
            i.e., [CLS], [dataset_token], and [SEP]
            '''
            indices = [0] + ([1] if self.hparams.include_dataset_token else
                             []) + indices + [input_len + offset]
            char_node_indices.append(indices)
        return char_node_indices

    def _get_char_wv_mats(self, batch_lattice):
        vector_lookup_fn = self._get_wv_model().get_word_vector
        char_node_ids = self._get_char_node_ids_from_lattice(batch_lattice)
        wv_mats = []
        for node_ids, tokens in zip(char_node_ids, batch_lattice.token):
            char_tokens = [tokens[id] for id in node_ids]
            wv_mat = torch.stack([
                torch.tensor(vector_lookup_fn(char_token), device=self.device)
                for char_token in char_tokens
            ])
            wv_mats.append(wv_mat)
        return pad_sequence(wv_mats)

    def _get_node_attrs_wv(self, batch_lattice):
        '''flatten tokens'''
        tokens = [token for tokens in batch_lattice.token for token in tokens]
        '''get wv_model and get vector lookup function'''
        vector_lookup_fn = self._get_wv_model().get_word_vector
        '''gather vectors from tokens'''
        wv_mat = torch.stack([
            torch.tensor(vector_lookup_fn(token), device=self.device)
            for token_idx, token in enumerate(tokens)
        ])
        return wv_mat

    def _replace_char_node_attrs(self, batch_lattice, batch, input_lengths):
        '''
        replace nodes by their corresponding char node ids
        - char_node_ids specifies the rows to be replaced in graph
            (batch-level)
        - char_node_indices specifies the rows to be accessed in bert-output
            (batch-level)
        '''
        '''get char_node_ids for batch_lattice'''
        char_node_ids = self._get_char_node_ids_from_lattice(batch_lattice)
        '''get char_node_ids for batch'''
        char_node_indices = self._get_char_node_indices(
            batch_lattice, batch, input_lengths)
        _batch_lattice = batch_lattice.detach().clone()
        for i, (node_ids, node_indices, data) in enumerate(
                zip(char_node_ids, char_node_indices,
                    _batch_lattice.to_data_list())):
            data.x[node_ids] = batch[i][node_indices].detach().clone()
        return _batch_lattice

    def _concat_outputs_with_wv(self, outputs: torch.Tensor,
                                batch_lattice: torch.Tensor) -> torch.Tensor:
        char_wv_mats = self._get_char_wv_mats(batch_lattice)
        return torch.cat([outputs, char_wv_mats], dim=2)

    def _concat_node_attrs_with_wv(
            self, batch_lattice: torch.Tensor) -> torch.Tensor:
        wv_mat = self._get_node_attrs_wv(batch_lattice)
        _batch_lattice = batch_lattice.detach().clone()
        node_attrs = _batch_lattice.x.detach().clone()
        _batch_lattice.x = torch.cat([node_attrs, wv_mat], dim=1)
        return _batch_lattice

    def configure_optimizers(self):
        parameters = self._optimizer_grouped_parameters()
        optimizer = torch.optim.AdamW(parameters, lr=self.hparams.lr)
        return_values = [optimizer]
        if self.hparams.scheduler:
            scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer, gamma=self.hparams.lr_decay_rate)
            return_values = ([optimizer], [scheduler])
        return return_values

    def _optimizer_grouped_parameters(self):
        optimizer_grouped_parameters = []

        # common
        uncommon = ['bert', 'gnn', 'classifier']
        common_params = [
            p for n, p in self.named_parameters()
            if not any(uc in n for uc in uncommon)
        ]
        optimizer_grouped_parameters.append({
            'params': common_params,
            'lr': self.hparams.lr
        })

        # bert
        optimizer_grouped_parameters.append({
            'params':
            self.classifier.parameters(),
            'lr':
            self.hparams.bert_lr
        })
        if self.hparams.optimized_decay:
            no_decay = ['bias', 'LayerNorm.weight']
            bert_decay_params = [
                p for n, p in self.bert.named_parameters()
                if not any(nd in n for nd in no_decay)
            ]
            bert_no_decay_params = [
                p for n, p in self.bert.named_parameters()
                if any(nd in n for nd in no_decay)
            ]
            optimizer_grouped_parameters.append({
                'params': bert_decay_params,
                'weight_decay': 0.01,
                'lr': self.hparams.bert_lr
            })
            optimizer_grouped_parameters.append({
                'params': bert_no_decay_params,
                'weight_decay': 0.0,
                'lr': self.hparams.bert_lr
            })
        else:
            optimizer_grouped_parameters.append({
                'params':
                self.bert.parameters(),
                'lr':
                self.hparams.bert_lr
            })

        # gnn
        optimizer_grouped_parameters.append({
            'params': self.gnn.parameters(),
            'lr': self.hparams.gnn_lr
        })

        return optimizer_grouped_parameters

    @staticmethod
    def add_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser],
                                         add_help=False)
        parser.add_argument('--embed-size', type=int, default=256)
        parser.add_argument('--hidden-size', type=int, default=300)
        parser.add_argument('--num-layers', type=int, default=2)
        parser.add_argument('--gnn-lr', type=float, default=1e-3)
        parser.add_argument('--gnn-type',
                            choices=['gcn', 'gat'],
                            default='gat')
        parser.add_argument('--gnn-num-heads', type=int, default=2)
        parser.add_argument('--unidirectional-gnn', action='store_true')
        parser.add_argument('--shallow-gnn', action='store_true')
        parser.add_argument('--graph-dropout', type=float, default=0.0)
        parser.add_argument('--attn-dropout', type=float, default=0.1)
        parser.add_argument('--max-token-length', type=int, default=4)
        parser.add_argument('--node-comp-type',
                            choices=['none', 'wv'],
                            default='none')
        parser.add_argument('--attn-comp-type',
                            choices=['wavg', 'wcon'],
                            default='wavg')
        parser.add_argument('--unuse-context-char-node', action='store_true')
        parser.add_argument('--wv-model-path', type=Path)
        parser.add_argument('--unuse-dynamic-graph', action='store_true')
        parser.add_argument('--generate-unigram-node', action='store_true')
        return parser
