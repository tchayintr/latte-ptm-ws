import copy
from collections import Counter
import fasttext
import logging
import math
from pathlib import Path
import pickle
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import transformers

from utils.tokenizer import construct_bert_tokenizer
from utils.data.dataset import (Dataset, BertDataset, augment_lines_with_token)
from utils import graph

# uncomment to debug
logging.basicConfig(level='DEBUG')
logger = logging.getLogger(__name__)


class DataModule(pl.LightningDataModule):

    def __init__(self, data_dir, ext_dic_file, wv_model_path, batch_size,
                 model_max_seq_length, max_token_length, lang,
                 normalize_unicode, seed, pretrained_model,
                 pretrained_save_path, shuffle_data, generate_unigram_node,
                 include_dataset_token, unc_token_ratio, include_unc_token,
                 include_lattice, build_dynamic_graph, include_valid_vocab,
                 train_split_ratio, node_comp_type, graph_dropout, use_binary):
        super(DataModule, self).__init__()
        self.data_dir = data_dir
        self.ext_dic_file = ext_dic_file
        self.wv_model_path = wv_model_path
        self.batch_size = batch_size
        self.model_max_seq_length = model_max_seq_length
        self.max_token_length = max_token_length
        self.lang = lang
        self.normalize_unicode = normalize_unicode
        self.seed = seed
        self.pretrained_model = pretrained_model
        self.pretrained_save_path = pretrained_save_path
        self.shuffle_data = shuffle_data
        self.generate_unigram_node = generate_unigram_node
        self.include_dataset_token = include_dataset_token
        self.unc_token_ratio = unc_token_ratio
        self.include_unc_token = include_unc_token
        self.include_lattice = include_lattice
        self.build_dynamic_graph = build_dynamic_graph
        self.include_valid_vocab = include_valid_vocab
        self.train_split_ratio = train_split_ratio
        self.node_comp_type = node_comp_type
        self.graph_dropout = graph_dropout
        self.use_binary = use_binary

        self.train_data = None
        self.valid_data = None

        self.train_set = None
        self.valid_set = None

        self._train_vocab = None
        self.vocab = None

        self.tokenizer = None
        self.tokenizer_padding_idx = None
        self.trie = None
        self.ext_vocab_data = None
        self.dataset_tokens = []
        self.unc_token = '[UNC]'

        self.wv_model = None

    def setup(self, stage=None):
        if self.use_binary and self._binary_file_path_exists():
            self._setup_data_from_binary()
            self._setup_tokenizer()
            if self.node_comp_type == 'wv':
                self._setup_wv_model()
        else:
            self._setup_train_and_valid_data()
            self._setup_ext_vocab_data()
            self._setup_trie()
            self._setup_vocab()
            self._setup_tokenizer()
            if self.node_comp_type == 'wv':
                self._setup_wv_model()

            params = self._get_bert_dataset_params(tokenizer=self.tokenizer)
            self.train_set = BertDataset(data=self.train_data,
                                         train=True,
                                         **params)
            self.valid_set = BertDataset(data=self.valid_data,
                                         train=False,
                                         **params)

            if self.use_binary:
                self._save_data_to_binary()

    def _setup_train_and_valid_data(self):
        '''
        Load datasets, normalise lines, split lines into list,
        augment (add) a dataset token, e.g., [BCCWJ]  to each line,
        and update dataset token list (self.dataset_tokens)
        '''
        data_path = Path(self.data_dir).glob('**/*')
        file_paths = [f for f in data_path if f.is_file()]

        raw_data = {}
        for file_path in file_paths:
            dataset_name = file_path.stem.split('.')[0]
            raw_data[dataset_name] = Dataset.read_dataset(file_path)

        data = []
        for dataset_name in raw_data:
            norm_lines = [(Dataset.normalize_line(line, self.lang)
                           if self.normalize_unicode else line).split()
                          for line in raw_data[dataset_name]]
            if self.include_dataset_token:
                dataset_token = '[' + dataset_name.upper() + ']'
                norm_lines = augment_lines_with_token(
                    norm_lines,
                    dataset_token,
                    index=0,
                    include_unc_token=self.include_unc_token,
                    unc_token=self.unc_token,
                    unc_token_ratio=(self.unc_token_ratio
                                     if self.include_unc_token else 0.0))
                self._update_dataset_token(dataset_token)
            data.extend(norm_lines)

        if self.train_split_ratio > 1.0 or self.train_split_ratio < 0.0:
            raise AssertionError('0.0 > train-split-ratio < 1.0')

        self.train_data, self.valid_data = train_test_split(
            data,
            train_size=self.train_split_ratio,
            random_state=self.seed,
            shuffle=self.shuffle_data)

    def _setup_ext_vocab_data(self):
        if self.ext_dic_file:
            ext_vocab_data = Dataset.read_dataset(self.ext_dic_file)
            ext_vocab_data = [(Dataset.normalize_line(line, self.lang)
                               if self.normalize_unicode else line)
                              for line in ext_vocab_data]
            self.ext_vocab_data = sorted(set(ext_vocab_data))

    def _setup_trie(self):
        data = copy.deepcopy(self.train_data)

        if self.include_valid_vocab:
            if self.valid_data is None:
                raise AssertionError(
                    '`--include-valid-vocab` requires `--train-split-ratio`')
            data.extend(self.valid_data)

        if self.ext_vocab_data is not None:
            data.extend(self.ext_vocab_data)

        token_counter = Counter()
        for s in data:
            token_counter.update(s)
            if self.generate_unigram_node:
                is_ext_vocab = isinstance(s, str)
                if (self.include_dataset_token and not is_ext_vocab):
                    s = s[1:]

                unigrams = list(''.join(s))
                token_counter.update(unigrams)

        trie = graph.Trie()
        for token in tqdm(
                token_counter.keys(),
                desc='Construct Trie(data={}, ext_vocab_data={})'.format(
                    self.train_data is not None, self.ext_vocab_data
                    is not None)):
            trie.add_token(token)
        trie._build_trie()
        self.trie = trie

    def _setup_vocab(self):
        '''
        Only using unigram tokens, no any valid/test gold (word) token
        _train_vocab is used for evaluating oov-recall
        '''
        token_counter = Counter()
        for s in tqdm(self.train_data,
                      desc='Setup vocab(train={})'.format(
                          self.train_data is not None)):
            if self.include_dataset_token:
                s = s[1:]
            token_counter.update(s)

        params = self._get_vocab_params()
        self._train_vocab = Dataset.construct_vocab(
            tokens=token_counter.keys(), **params)

        if (self.valid_data is not None and self.include_valid_vocab):
            for s in self.valid_data:
                if self.include_dataset_token:
                    s = s[1:]
                token_counter.update(s)

        if self.ext_vocab_data is not None:
            for s in self.ext_vocab_data:
                token_counter.update(s)

        data = self.train_data + self.valid_data + (
            self.ext_vocab_data if self.ext_vocab_data is not None else [])
        for s in tqdm(
                data,
                desc='Setup vocab(train={}, valid={}, ext_vocab={}, trie={})'.
                format(self.train_data is not None, self.valid_data
                       is not None, self.ext_vocab_data is not None, self.trie
                       is not None)):
            is_ext_vocab = isinstance(s, str)
            if (self.include_dataset_token and not is_ext_vocab):
                s = s[1:]
            unigrams = list(''.join(s))
            token_counter.update(unigrams)
            tokens = [
                token for token, span in self.trie.search_tokens_from_trie(s)
            ]
            token_counter.update(tokens)

        self.vocab = Dataset.construct_vocab(tokens=token_counter.keys(),
                                             **params)

    def _setup_tokenizer(self):
        '''
        Setup tokenizer based on pretrained model
        '''
        '''setup tokenizer from binary file'''
        tokenizer_bin_path = Path(self.pretrained_model) / 'tokenizer.pkl'
        '''the model/tokenizer have been trained before'''
        tokenizer_bin_save_path = self.pretrained_save_path / 'tokenizer.pkl'
        if tokenizer_bin_path.exists():
            logger.info(f'Load binary tokenizer: {tokenizer_bin_path}')
            with open(tokenizer_bin_path, 'rb') as f:
                self.tokenizer = pickle.load(f)
        elif tokenizer_bin_save_path.exists():
            '''setup tokenizer from binary file'''
            logger.info(f'Load binary tokenizer: {tokenizer_bin_save_path}')
            with open(tokenizer_bin_save_path, 'rb') as f:
                self.tokenizer = pickle.load(f)
        else:
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(
                self.pretrained_model, use_fast=False)

            special_tokens = {
                'pad_token': '[PAD]',
                'unk_token': '[UNK]',
                'mask_token': '[MASK]',
                'cls_token': '[CLS]',
                'sep_token': '[SEP]',
                'bos_token': '[BOS]',
                'eos_token': '[EOS]',
            }
            additional_special_tokens = [self.unc_token] + self.dataset_tokens

            vocab = copy.deepcopy(self.vocab)
            '''filter by length'''
            vocab = [
                token for token in vocab if len(token) <= self.max_token_length
                and token not in additional_special_tokens
            ]

            vocab = sorted(vocab)
            '''add special tokens'''
            self.tokenizer.add_special_tokens(special_tokens)
            self.tokenizer.add_special_tokens(
                {"additional_special_tokens": additional_special_tokens})
            '''
            add new common tokens
            batching is required becuase of the limitation of add_tokens
            '''
            vocab_batch = self.batch_iterator(vocab, batch_size=6400)
            vocab_batch_size = 6400
            for b in tqdm(
                    vocab_batch,
                    desc='Tokenizer(add_tokens=True, vocab_batch_size={})'.
                    format(vocab_batch_size),
                    total=math.ceil(len(vocab) / vocab_batch_size)):
                self.tokenizer.add_tokens(b)
            '''save pretrained tokenizer'''
            self.tokenizer.save_pretrained(self.pretrained_save_path)
            '''save pretrained tokenizer as a binary file'''
            tokenizer_bin_save_path = (self.pretrained_save_path /
                                       'tokenizer.pkl')
            with open(tokenizer_bin_save_path, 'wb') as f:
                logger.info(
                    f'Save binary tokenizer: {tokenizer_bin_save_path}')
                pickle.dump(self.tokenizer,
                            f,
                            protocol=pickle.HIGHEST_PROTOCOL)
        '''update tokenizer padding index'''
        self.tokenizer_padding_idx = self.tokenizer.convert_tokens_to_ids(
            '[PAD]')
        if self.tokenizer_padding_idx is None:
            raise AssertionError

    def _setup_new_tokenizer(self):
        '''
        Setup tokenizer if tokenizer (json) is provided,
        otherwise, train a new tokenizer
        '''

        if self.tokenizer_path is not None:
            self.tokenizer = transformers.BertTokenizer.from_pretrained(
                self.tokenizer_path, use_fast=False)
        else:
            special_tokens = ['[PAD]', '[UNK]', '[MASK]', '[CLS]', '[SEP]']
            additional_special_tokens = [
                '[BOS]',
                '[EOS]',
            ] + [self.unc_token] + self.dataset_tokens
            '''extract unigrams'''
            vocab = copy.deepcopy(set(self.vocab))
            for token in self.vocab:
                if len(token) > 1 and (token not in special_tokens and token
                                       not in additional_special_tokens
                                       and token not in self.dataset_tokens):
                    '''update method adds unigram'''
                    vocab.update(token)
            vocab = sorted(vocab)
            handle_chinese_chars = (self.lang == 'zh' or self.lang == 'ja')

            self.tokenizer = construct_bert_tokenizer(
                vocab=vocab,
                handle_chinese_chars=handle_chinese_chars,
                pad_token='[PAD]',
                unk_token='[UNK]',
                mask_token='[MASK]',
                cls_token='[CLS]',
                sep_token='[SEP]',
                additional_special_tokens=additional_special_tokens,
                save_path=str(self.tokenizer_save_path),
                return_tokenizer=True)

    def _setup_wv_model(self):
        if self.wv_model_path is None:
            raise AssertionError(
                ' `--wv-model-path` is required for `--node_comp_type==wv`')
        wv_model_path_str = str(self.wv_model_path.resolve())
        self.wv_model = fasttext.load_model(wv_model_path_str)

    def _setup_data_from_binary(self):
        (self.train_data, self.valid_data, self.ext_vocab_data, self.train_set,
         self.valid_set, self.trie, self.vocab, self._train_vocab,
         self.dataset_tokens) = self._load_data_from_binary()

    def _get_binary_file_path(self) -> Path:
        return self.pretrained_save_path / 'data.pkl'

    def _load_data_from_binary(self) -> Dataset:
        bin_file_path = self._get_binary_file_path()
        logger.info(f'Load binary data: {bin_file_path}')
        with open(bin_file_path, 'rb') as f:
            bin_data = pickle.load(f)
            train_data = bin_data['train_data']
            valid_data = bin_data['valid_data']
            ext_vocab_data = bin_data['ext_vocab_data']
            train_set = bin_data['train_set']
            valid_set = bin_data['valid_set']
            trie = bin_data['trie']
            vocab = bin_data['vocab']
            _train_vocab = bin_data['_train_vocab']
            dataset_tokens = bin_data['dataset_tokens']
        return (train_data, valid_data, ext_vocab_data, train_set, valid_set,
                trie, vocab, _train_vocab, dataset_tokens)

    def _save_data_to_binary(self) -> bool:
        bin_file_path = self._get_binary_file_path()
        logger.info(f'Save binary data: {bin_file_path}')
        if not bin_file_path.exists():
            with open(bin_file_path, 'wb') as f:
                obj = dict({
                    'train_data': self.train_data,
                    'valid_data': self.valid_data,
                    'ext_vocab_data': self.ext_vocab_data,
                    'train_set': self.train_set,
                    'valid_set': self.valid_set,
                    'trie': self.trie,
                    'vocab': self.vocab,
                    '_train_vocab': self._train_vocab,
                    'dataset_tokens': self.dataset_tokens,
                })
                pickle.dump(obj, f)
                return True
        return False

    def _binary_file_path_exists(self):
        binary_file_path = self._get_binary_file_path()
        if binary_file_path.exists():
            return True
        return False

    def _get_bert_dataset_params(self, tokenizer=None):
        return dict({
            'vocab_org': self.vocab,
            'tokenizer': tokenizer,
            'model_max_seq_length': self.model_max_seq_length,
            'lang': self.lang,
            'normalize_unicode': self.normalize_unicode,
            'max_token_length': self.max_token_length,
            'trie_org': self.trie,
            'include_dataset_token': self.include_dataset_token,
            'include_lattice': self.include_lattice,
            'node_comp_type': self.node_comp_type,
            'graph_dropout': self.graph_dropout,
            'use_custom_encoder': True,
        })

    def _get_vocab_params(self):
        return dict({
            'pad_token': '[PAD]',
            'unk_token': '[UNK]',
            'bos_token': '[BOS]',
            'eos_token': '[EOS]',
            'cls_token': '[CLS]',
            'sep_token': '[SEP]',
            'unc_token': '[UNC]',
            'dataset_tokens': self.dataset_tokens,
            'padding_idx': 0,
            'unk_idx': 1,
            'bos_idx': 2,
            'eos_idx': 3,
            'cls_idx': 4,
            'sep_idx': 5,
            'unc_idx': 6,
            'sort_token': True,
        })

    def _update_dataset_token(self, token: str) -> None:
        if token in self.dataset_tokens:
            raise AssertionError('Duplicate dataset detected')
        self.dataset_tokens.append(token)
        self.dataset_tokens = sorted(set(self.dataset_tokens))

    '''
    https://huggingface.co/docs/tokenizers/python/latest/
    tutorials/python/training_from_memory.html
    '''

    @staticmethod
    def batch_iterator(data, batch_size: int = 32):
        for i in range(0, len(data), batch_size):
            yield data[i:i + batch_size]
