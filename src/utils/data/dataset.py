import abc
from collections import Counter
import copy
import logging
from pathlib import Path
import pythainlp
import random
import torch
from torch_geometric.data import Data as GraphData
from torch_geometric.data import Batch as GraphBatch
from tqdm import tqdm
from typing import Any, List, Dict, Sequence, Tuple, Optional, Union
import unicodedata


from utils import graph

# uncomment to debug
logging.basicConfig(level='DEBUG')
logger = logging.getLogger(__name__)

PTM_MAX_LENGTH_LIMIT = 512
'''
padding_idx = 0 if it is related to vocab id
padding_idx = -1 if it is related to label id
'''


class Dataset(torch.utils.data.Dataset):
    def __init__(self,
                 data: List,
                 vocab_org: Dict,
                 lang: str,
                 normalize_unicode=False,
                 max_token_length: int = 4,
                 trie_org=None,
                 include_dataset_token: bool = None,
                 include_lattice: bool = False,
                 node_comp_type: str = 'none',
                 graph_dropout: float = 0.0,
                 build_dynamic_graph: bool = False,
                 generate_unigram_node: bool = False,
                 train=False):
        self.data = data
        self.vocab_org = vocab_org
        self.lang = lang
        self.normalize_unicode = normalize_unicode
        self.max_token_length = max_token_length
        self.trie_org = trie_org
        self.include_dataset_token = include_dataset_token
        self.include_lattice = include_lattice
        self.node_comp_type = node_comp_type
        self.graph_dropout = graph_dropout
        self.build_dynamic_graph = build_dynamic_graph
        self.generate_unigram_node = generate_unigram_node

        self.train = train

        self.bmes_dict = {'B': 0, 'M': 1, 'E': 2, 'S': 3}

        self.features = []
        self.labels = []

        self.trie = None

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        x = self.features[index]
        y = self.labels[index]
        return x, y

    def process_data(self) -> Tuple[List[str], List[str]]:
        x = [
            self.process_tokens(s)
            for s in tqdm(self.data,
                          desc='Process tokens(train={})'.format(self.train))
        ]
        y = [
            self._convert_labels(s)
            for s in tqdm(self.data,
                          desc='Process labels(train={})'.format(self.train))
        ]
        return x, y

    def _tokens2unigrams(self, tokens: List[str]) -> List[str]:
        return list(''.join(tokens))

    def _tokens2text(self,
                     tokens: List[str],
                     max_length: Optional[int] = None) -> str:
        if max_length:
            return ''.join(tokens)[:max_length]
        return ''.join(tokens)

    @abc.abstractmethod
    def process_tokens(self, tokens: List, max_length: int) -> Any:
        pass

    @abc.abstractmethod
    def _convert_labels(self, tokens: List, max_length: int) -> Any:
        pass

    @abc.abstractmethod
    def _text2lattice(self, text: str):
        pass

    @staticmethod
    def normalize_tokens(tokens: List[str], lang: str = 'none') -> List[str]:
        if lang == 'zh' or lang == 'ja':
            return [half_width_char(token) for token in tokens]
        elif lang == 'th':
            return [pythainlp.util.normalize(token) for token in tokens]
        return tokens

    @staticmethod
    def normalize_line(line: str, lang: str = 'none') -> str:
        if lang == 'zh' or lang == 'ja':
            return half_width_char(line)
        elif lang == 'th':
            return pythainlp.util.normalize(line)
        return line

    @staticmethod
    def read_dataset(path: Path) -> List[str]:
        mode = 'rb' if path.suffix == '.bin' else 'r'
        with open(path, mode=mode) as file:
            f = (line.strip() for line in tqdm(file, desc=f'Read file {path}'))
            return [line for line in f if line]

    @staticmethod
    def construct_vocab(tokens: List[str],
                        pad_token: str = '<pad>',
                        unk_token: str = '<unk>',
                        bos_token: str = '<cls>',
                        eos_token: str = '<sep>',
                        cls_token: Optional[str] = None,
                        sep_token: Optional[str] = None,
                        unc_token: Optional[str] = None,
                        dataset_tokens: Optional[List[str]] = None,
                        padding_idx: int = 0,
                        unk_idx: int = 1,
                        bos_idx: int = 2,
                        eos_idx: int = 3,
                        cls_idx: Optional[int] = 4,
                        sep_idx: Optional[int] = 5,
                        unc_idx: Optional[int] = 6,
                        sort_token: bool = True) -> Dict[str, int]:
        vocab = {
            pad_token: padding_idx,
            unk_token: unk_idx,
            bos_token: bos_idx,
            eos_token: eos_idx
        }
        if cls_token is not None:
            vocab[cls_token] = cls_idx
            vocab_idx = cls_idx
        if sep_token is not None:
            vocab[sep_token] = sep_idx
            vocab_idx = sep_idx
        if unc_token is not None:
            vocab[unc_token] = unc_idx
            vocab_idx = unc_idx

        max_idx = max(vocab.values())
        vocab_idx = max_idx
        if dataset_tokens is not None:
            for dtoken in dataset_tokens:
                vocab[dtoken] = vocab_idx
                vocab_idx += 1

        if sort_token:
            tokens = sorted(tokens)
        for token in tokens:
            if token not in vocab.keys():
                vocab[token] = vocab_idx
                vocab_idx += 1
        return vocab

    @staticmethod
    def extend_trie(trie_org, data: List, normalize_unicode: bool, lang: str):
        token_counter = Counter()
        for s in data:
            unigrams = list(''.join(s))
            token_counter.update(unigrams)

        trie = copy.deepcopy(trie_org)
        for s in tqdm(token_counter.keys(),
                      desc='Extend Trie(data={})'.format(len(data))):
            trie.add_token(s)
        trie._build_trie()
        return trie


class BertDataset(Dataset):
    def __init__(self,
                 data: List,
                 vocab_org: Dict,
                 tokenizer,
                 model_max_seq_length: int = -1,
                 lang: str = 'none',
                 normalize_unicode: bool = False,
                 max_token_length: int = 4,
                 trie_org=None,
                 include_dataset_token: bool = False,
                 include_lattice: bool = False,
                 node_comp_type: str = 'none',
                 build_dynamic_graph: bool = True,
                 graph_dropout: float = 0.0,
                 generate_unigram_node: bool = True,
                 use_custom_encoder: bool = True,
                 train: bool = False):
        super().__init__(data, vocab_org, lang, normalize_unicode,
                         max_token_length, trie_org, include_dataset_token,
                         include_lattice, node_comp_type, graph_dropout,
                         build_dynamic_graph, generate_unigram_node, train)
        self.tokenizer = tokenizer
        self.model_max_seq_length = model_max_seq_length
        self.use_custom_encoder = use_custom_encoder
        self.trie = self.extend_trie(
            trie_org, data, normalize_unicode, lang,
            include_dataset_token) if ((not train and trie_org)
                                       and generate_unigram_node) else trie_org
        self.features, self.labels = self.process_data()

    def process_tokens(self, tokens: List) -> Dict[str, Sequence[int]]:
        dataset_token = None
        len_offset = 2
        if self.include_dataset_token:
            dataset_token = tokens[0]
            len_offset = 3
            tokens = tokens[1:]

        unigrams = self._tokens2unigrams(tokens)
        unigram_len = len(unigrams)
        _input_seq_length_for_model = self._get_input_seq_length_for_model(
            unigram_len)
        truncated = self._is_truncated(unigram_len,
                                       _input_seq_length_for_model)
        if truncated:
            unigrams = unigrams[:_input_seq_length_for_model - len_offset]
            unigram_len = len(unigrams)

        if self.use_custom_encoder:
            char_inputs = self.encode_sequence(
                unigrams,
                self.tokenizer,
                max_length=_input_seq_length_for_model,
                add_special_tokens=True,
                dataset_token=dataset_token,
                is_split_into_words=True,
                truncated=True,
                return_attention_mask=True)
        else:
            char_inputs = self.tokenizer(
                tokens,
                max_length=_input_seq_length_for_model,
                add_special_tokens=True,
                is_split_into_words=True,
                padding=True,
                truncation=True,
                return_attention_mask=True)
            char_inputs = char_inputs.data
            if self.include_dataset_token:
                dataset_token_id = self.tokenizer.convert_tokens_to_ids(
                    dataset_token)
                char_inputs['input_ids'].insert(1, dataset_token_id)
                char_inputs['token_type_ids'].insert(1, 0)
                char_inputs['attention_mask'].insert(1, 1)

        org_ids = tokens2ids(unigrams, self.vocab_org)
        text = self._tokens2text(tokens, unigram_len)

        lattice, mapping = None, None
        lattice_params, mapping_params = None, None
        if self.include_lattice:
            lattice_params = self._get_lattice_params(
                dataset_token=dataset_token)
            mapping_params = self._get_mapping_params(
                dataset_token=dataset_token)
            lattice = self.text2lattice(text, **lattice_params)
            mapping = construct_lattice_mapping(unigrams, lattice,
                                                **mapping_params)
        return dict({
            'char':
            char_inputs,
            'org_ids':
            org_ids,
            'lattice':
            lattice,
            'mapping':
            mapping,
            'lattice_params':
            (lattice_params if self.build_dynamic_graph else None),
            'mapping_params':
            (mapping_params if self.build_dynamic_graph else None),
        })

    def _convert_labels(self, tokens: List) -> Sequence[int]:
        '''
        [3] -> 'S', [-1] -> padding index to be ignored
        '''
        len_offset = 2
        if self.include_dataset_token:
            len_offset = 3
            tokens = tokens[1:]

        labels = tokens2labels(tokens)
        _input_seq_length_for_model = self._get_input_seq_length_for_model(
            len(labels))
        converted_line = [
            self.bmes_dict[label]
            for label in labels[:_input_seq_length_for_model - len_offset]
        ]
        converted_line = [3] + ([3] if self.include_dataset_token else
                                []) + converted_line + [3]
        if _input_seq_length_for_model > len_offset:
            converted_line += [-1] * (_input_seq_length_for_model -
                                      len(converted_line))
        return converted_line

    @staticmethod
    def text2lattice(text: str,
                     trie,
                     tokenizer,
                     vocab: Dict[int, str],
                     pad_token: str = '[PAD]',
                     unk_token: str = '[UNK]',
                     bos_token: str = '[BOS]',
                     eos_token: str = '[EOS]',
                     dataset_token: Optional[str] = None,
                     max_token_length: int = 4,
                     node_comp_type: str = 'none',
                     rand_dropout: float = 0.0,
                     keep_unigram_node: bool = False,
                     directional: str = 'bidirectional') -> GraphData:
        g = graph.BertLattice(tokens=text,
                              trie=trie,
                              tokenizer=tokenizer,
                              vocab=vocab,
                              pad_token=pad_token,
                              unk_token=unk_token,
                              bos_token=bos_token,
                              eos_token=eos_token,
                              dataset_token=dataset_token,
                              max_token_length=max_token_length,
                              node_comp_type=node_comp_type,
                              rand_dropout=rand_dropout,
                              keep_unigram_node=keep_unigram_node,
                              directional=directional)._get_pyg_graph()
        return g

    def _is_truncated(self, token_len: int, model_seq_length: int) -> bool:
        len_offset = (3 if self.include_dataset_token else 2)
        return False if token_len + len_offset <= model_seq_length else True

    def _get_input_seq_length_for_model(self, token_len: int) -> int:
        token_offset = 3 if self.include_dataset_token else 2
        if self.model_max_seq_length < (
                3 + (1 if self.include_dataset_token else 0)):
            '''
            dynamic length:
                If token_len + 2 or 3 ([CLS], [SEP], [UNC]) is in limitation,
                use token_len + 2 or 3, otherwise, use maximum limitation,
                and shink tokens later at least 2 or 3 original tokens
                for [CLS], [SEP], and [UNC].
            '''
            input_seq_length = (
                token_len + token_offset) if token_len + token_offset <= (
                    PTM_MAX_LENGTH_LIMIT) else PTM_MAX_LENGTH_LIMIT
        elif self.model_max_seq_length <= PTM_MAX_LENGTH_LIMIT:
            '''
            fixed length:
                If use fixed length and it is in limitation,
                use the token_len, otherwise, use the fixed length.
            '''
            input_seq_length = (token_len + token_offset if token_len +
                                token_offset <= self.model_max_seq_length else
                                self.model_max_seq_length)
        else:
            raise ValueError

        return input_seq_length

    def _get_lattice_params(self, dataset_token: Optional[str] = None):
        return dict({
            'trie': self.trie.get_automaton_trie(),
            'tokenizer': self.tokenizer,
            'vocab': self.vocab_org,
            'pad_token': '[PAD]',
            'unk_token': '[UNK]',
            'bos_token': '[BOS]',
            'eos_token': '[EOS]',
            'dataset_token': dataset_token,
            'max_token_length': self.max_token_length,
            'node_comp_type': self.node_comp_type,
            'rand_dropout': self.graph_dropout,
            'keep_unigram_node': self.generate_unigram_node,
            'directional': 'bidirectional'
        })

    def _get_mapping_params(self, dataset_token: Optional[str] = None):
        return dict({
            'max_token_length': self.max_token_length,
            'padding_value': '[PAD]',
            'bos_token': '[BOS]',
            'eos_token': '[EOS]',
            'dataset_token': dataset_token,
            'padding_node_idx': 0,
        })

    def _insert_dataset_token_id(inputs, idx: int, dataset_token_id: int):
        inputs['input_ids'].insert(idx, dataset_token_id)
        inputs['token_type_ids'].insert(idx, 0)
        inputs['attention_mask'].insert(idx, 1)

    @staticmethod
    def generate_batch(
        batch: Dict,
        vocab: Dict[str, int],
        include_lattice: bool = False,
        build_dynamic_graph: bool = False,
        dynamic_graph_dropout: float = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        generate batch for DataLoader
        :param batch: batch to process
        :param vocab: token -> token_id (to be used for generating unigram)
        :param include_lattice: to include_lattice in generated batch
        :param build_dynamic_graph:
            to rebuild lattice dynamically using dynamic_graph_dropout
        :param dynamic_graph_dropout:
            dropout rate for building dynamic graph
        :return: tokens and labels to feed to the model
        '''

        input_ids = pad_sequence(
            [torch.tensor(b[0]['char']['input_ids']) for b in batch])
        attention_mask = pad_sequence(
            [torch.tensor(b[0]['char']['attention_mask']) for b in batch])
        token_type_ids = pad_sequence(
            [torch.tensor(b[0]['char']['token_type_ids']) for b in batch])
        org_ids = pad_sequence([torch.tensor(b[0]['org_ids']) for b in batch])
        labels = pad_sequence([torch.tensor(b[1]) for b in batch],
                              padding_idx=-1)
        lattice, mapping = None, None
        if include_lattice:
            '''build dynamic graph (random remove nodes for each batch)'''
            if (build_dynamic_graph and dynamic_graph_dropout > 0.0):
                lattice_batch, mapping_batch = [], []
                for b in batch:
                    lattice_params = b[0]['lattice_params']
                    mapping_params = b[0]['mapping_params']
                    if (lattice_params is None or mapping_params is None):
                        raise AssertionError

                    text = ids2tokens(b[0]['org_ids'], vocab, padding_idx=0)
                    unigrams = list(text)
                    lattice = BertDataset.text2lattice(text=text,
                                                       **lattice_params)
                    mapping = construct_lattice_mapping(unigrams=unigrams,
                                                        lattice=lattice,
                                                        **mapping_params)
                    lattice_batch.append(lattice)
                    mapping_batch.append(torch.tensor(mapping))
                lattice = GraphBatch.from_data_list(lattice_batch)
                mapping = pad_sequence(mapping_batch)
            else:
                lattice = GraphBatch.from_data_list(
                    [b[0]['lattice'] for b in batch])
                mapping = pad_sequence(
                    [torch.tensor(b[0]['mapping']) for b in batch])

        features = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids,
            'org_ids': org_ids,
            'lattice': lattice,
            'mapping': mapping,
        }
        return features, labels

    @staticmethod
    def encode_sequence(text: Union[List[str], str],
                        tokenizer,
                        max_length: int,
                        add_special_tokens: bool = True,
                        dataset_token: Optional[str] = None,
                        is_split_into_words: bool = True,
                        truncated: bool = True,
                        return_attention_mask: bool = True) -> Dict:
        '''text input must not be included dataset_token'''
        if is_split_into_words:
            if not isinstance(text, list):
                raise AssertionError('`text` must be list')
        if isinstance(text, str):
            text = list(text)

        input_len = len(text)
        len_offset = 0
        if dataset_token is not None:
            if not isinstance(dataset_token, str):
                raise AssertionError
            len_offset += 1

        if truncated:
            if add_special_tokens:
                len_offset += 2
            truncated_max_length = max_length - len_offset
            if input_len > truncated_max_length:
                text = text[:truncated_max_length]
                input_len = len(text)

        pad_token_len = max_length - input_len
        input_ids = [tokenizer.convert_tokens_to_ids(token) for token in text]
        if dataset_token is not None:
            input_ids = [tokenizer.convert_tokens_to_ids(dataset_token)
                         ] + input_ids
        if add_special_tokens:
            input_ids = [
                tokenizer.convert_tokens_to_ids('[CLS]')
            ] + input_ids + [tokenizer.convert_tokens_to_ids('[SEP]')]
        pad_token_len -= len_offset
        input_ids = input_ids + ([tokenizer.convert_tokens_to_ids('[PAD]')] *
                                 pad_token_len)

        token_type_ids = [0] * max_length
        if return_attention_mask:
            attention_len = input_len
            if add_special_tokens:
                attention_len = input_len + len_offset
            attention_mask = [1 for _ in range(attention_len)
                              ] + ([0] * (pad_token_len))

        return dict({
            'input_ids': input_ids,
            'token_type_ids': token_type_ids,
            'attention_mask': attention_mask
        })

    @staticmethod
    def extend_trie(trie_org, data: List, normalize_unicode: bool, lang: str,
                    include_dataset_token: bool):
        token_counter = Counter()
        for s in data:
            if include_dataset_token:
                s = s[1:]
            unigrams = list(''.join(s))
            token_counter.update(unigrams)

        trie = copy.deepcopy(trie_org)
        for s in tqdm(token_counter.keys(),
                      desc='Extend Trie(data={})'.format(len(data))):
            trie.add_token(s)
        trie._build_trie()
        return trie


def tokens2ids(tokens: List[str],
               vocab: Dict[str, int],
               unk_token: str = '<unk>') -> List[int]:
    return [
        vocab[token] if token in vocab else vocab[unk_token]
        for token in tokens
    ]


def ids2tokens(ids: List, vocab: Dict[str, int], padding_idx: int = 0) -> str:
    vocab = {v: k for k, v in vocab.items()}
    return ''.join([vocab[id] for id in ids if id != padding_idx])


def tokens2labels(tokens: List) -> List:
    labels = []
    for token in tokens:
        if len(token) == 1:
            labels.append('S')
        else:
            labels.extend(['B'] + ['M'] * (len(token) - 2) + ['E'])
    return labels


def ids2chunks(ids: List[int],
               seq: str,
               constraint_type: str = 'BMES') -> List[str]:
    chunks = []
    if len(ids) == 0:
        return chunks

    bmes_dict = {0: 'B', 1: 'M', 2: 'E', 3: 'S'}
    chunk = seq[0]
    for c, id in zip(seq[1:], ids[1:]):
        label = bmes_dict[id]
        if label == 'B' or label == 'S':
            chunks.append(chunk)
            chunk = ''
        chunk += c
    if len(chunk) != 0:
        chunks.append(chunk)
    return chunks


def lattice2cspans(lattice: GraphData):
    '''convert lattice into char spans list'''
    spans = lattice.span.tolist()
    cspans = [tuple(span) for span in spans if span[1] - span[0] == 1]
    return cspans


def pad_sequence(data: List[torch.Tensor],
                 batch_first: bool = True,
                 padding_idx: int = 0,
                 include_length: bool = False):
    padded = torch.nn.utils.rnn.pad_sequence(data,
                                             batch_first=batch_first,
                                             padding_value=padding_idx)
    if include_length:
        lengths = torch.tensor([len(x) for x in data], dtype=torch.long)
        return padded, lengths
    return padded


def half_width_char(line: str) -> str:
    """
    Replaces full-width characters by half-width ones.
    :param line: string to process
    :return: the input string with half-width character.
    """
    return unicodedata.normalize('NFKC', line)


def construct_lattice_mapping(unigrams: List,
                              lattice: GraphData,
                              max_token_length: int = 4,
                              padding_value: str = '[PAD]',
                              bos_token: Optional[str] = '[BOS]',
                              eos_token: Optional[str] = '[EOS]',
                              dataset_token: Optional[str] = None,
                              padding_node_idx: int = 0) -> List:
    '''
        Initialize mapping table by padding_node_idx
        - e.g., max_token_length=3, unigram_tokens='abcde'
        - unigram_len = len(unigram_tokens)
        - L = 6: 1 + 2 + 3
        - Mapping(padding_node_idx) = Matrix[unigram_len, L] == Matrix[5, 6]
            [
              0 0 0 0 0 0
              0 0 0 0 0 0
              0 0 0 0 0 0
              0 0 0 0 0 0
              0 0 0 0 0 0
            ]
            ->
            [
              unigram1: l1 l2-1 l2-2 l3-1 l3-2 l3-3
              unigram2: l1 l2-1 l2-2 l3-1 l3-2 l3-3
              unigram3: l1 l2-1 l2-2 l3-1 l3-2 l3-3
              unigram4: l1 l2-1 l2-2 l3-1 l3-2 l3-3
              unigram5: l1 l2-1 l2-2 l3-1 l3-2 l3-3
            ]

        - note:
            unigram1: l1: a
            unigram1: l2-1: 0
            unigram1: l2-2: ab
            unigram1: l3-1: 0
            unigram1: l3-2: 0
            unigram1: l3-3: abc
            unigram2: l1: b
            unigram2: l2-1: ab
            unigram2: l2-2: bc
            unigram2: l3-1: 0
            unigram2: l3-2: abc
            unigram2: l3-3: bcd
            unigram3: l1: c
            unigram3: l2-1: bc
            unigram3: l2-2: cd
            unigram3: l3-1: abc
            unigram3: l3-2: bcd
            unigram3: l3-3: cde
            unigram4: l1: d
            unigram4: l2-1: cd
            unigram4: l2-2: de
            unigram4: l3-1: bcd
            unigram4: l3-2: cde
            unigram4: l3-3: 0
            unigram4: l1: e
            unigram4: l2-1: de
            unigram4: l2-2: 0
            unigram4: l3-1: cde
            unigram4: l3-2: 0
            unigram4: l3-3: 0

        - Fill mapping table with id (index) in lattice
            - filter by max_token_length
    '''
    # unigrams
    if ((bos_token is None and eos_token is not None)
            or (bos_token is not None and eos_token is None)):
        raise ValueError('Either `bos_token` or `eos_token` cannot be None')
    unigram_tokens = unigrams
    '''offset for shifting when special_tokens are included'''
    base_idx_offset = 0

    special_tokens = [bos_token, eos_token]
    additional_special_tokens = [dataset_token]
    '''remove None for offset'''
    additional_special_tokens = [
        aspt for aspt in additional_special_tokens if aspt
    ]
    if all(special_tokens) and not additional_special_tokens:
        '''special_tokens has no None but dataset_token is None'''
        unigram_tokens = [bos_token] + unigram_tokens + [eos_token]
        base_idx_offset = 1
    elif all(special_tokens) and additional_special_tokens:
        unigram_tokens = [bos_token] + [dataset_token
                                        ] + unigram_tokens + [eos_token]
        base_idx_offset = 2
    elif not all(special_tokens) and additional_special_tokens:
        '''
        disregard additional_special_tokens if no special_tokens
        but additional_special_tokens
        '''
        additional_special_tokens = []
    elif not all(special_tokens) and not additional_special_tokens:
        pass
    else:
        raise NotImplementedError

    # lattice
    token_attrs = lattice.token
    span_attrs = lattice.span
    if token_attrs[padding_node_idx] != padding_value:
        raise ValueError(
            'Invalid `padding_node_idx` for {}'.format(padding_value))

    # mapping
    L = sum([i for i in range(1, max_token_length + 1)])
    mapping = [[padding_node_idx for _ in range(L)]
               for i in range(len(unigram_tokens))]
    '''
    Looping order: pad, bos, eos, and dataset_token (if specify), tokens
    Mapping order: bos, dataset_token (if specify), tokens, and eos
    '''
    for attr_idx, (token, span) in enumerate(zip(token_attrs, span_attrs)):
        idx_offset = base_idx_offset
        token_length_offset = 0  # length offset for special_tokens
        start, end = span
        token_length = end - start

        if token == padding_value:
            continue
        if ((start == end) and not all(special_tokens)):
            continue

        if (token in special_tokens or token in additional_special_tokens):
            token_length_offset = 1
            if token == bos_token:
                idx_offset = 1
            elif token == eos_token:
                idx_offset = 2 + len(additional_special_tokens)
            elif token in additional_special_tokens:
                idx_offset = 3 - len(additional_special_tokens)
        elif (token not in special_tokens
              and token not in additional_special_tokens):
            if start == end:
                continue
            if token_length > max_token_length:
                continue

        i = (start + token_length - 1) + idx_offset
        j = sum([x for x in range(token_length)])
        for _l in range(token_length + token_length_offset):
            mapping[i][j] = attr_idx
            i = i - 1
            j = j + 1

    return mapping


def construct_shared_length_graph(max_length: int = 512,
                                  padding_idx: int = 0,
                                  directional: str = 'bidirectional',
                                  device: str = 'cuda'):
    vs = [i for i in range(1, max_length + 1)]
    return graph.LinearLattice(
        vectices=vs, padding_idx=padding_idx,
        directional=directional)._get_pyg_graph().to(device)


def augment_lines_with_token(lines: List[str],
                             token: str,
                             index: int = 0,
                             include_unc_token: bool = False,
                             unc_token: str = '[UNC]',
                             unc_token_ratio: float = 0.0):
    return [
        augment_line_with_token(line,
                                token,
                                index,
                                include_unc_token=include_unc_token,
                                unc_token=unc_token,
                                unc_token_ratio=unc_token_ratio)
        for line in lines
    ]


def augment_line_with_token(line: List[str],
                            token: str,
                            index: int = 0,
                            include_unc_token: bool = False,
                            unc_token: str = '[UNC]',
                            unc_token_ratio: float = 0.0):
    if random.random() < unc_token_ratio:
        line.insert(index, unc_token)
    elif token is not None:
        line.insert(index, token)
    return line
