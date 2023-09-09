from __future__ import annotations
import ahocorasick
import networkx as nx
import random
import torch
from torch_geometric.utils import from_networkx
from typing import List, Dict, Tuple, Optional, Union

PTM_MAX_LENGTH_LIMIT = 512


class Trie(object):

    def __init__(self):
        self._trie = None
        self.tokens = []

    def add_token(self, token: str):
        self.tokens.append(token)

    def add_tokens(self, tokens: List[str]):
        for token in tokens:
            self.add_token(token)

    def get_automaton_trie(self):
        return self._trie

    def search_tokens_from_trie(
            self, tokens: Optional[str,
                                   List]) -> List[Tuple[str, Tuple[int, int]]]:
        if not self._trie:
            return []
        if isinstance(tokens, list):
            tokens = ''.join(tokens)

        found_tokens = []
        for (end_idx, (insert_order, token)) in self._trie.iter(tokens):
            start_idx = end_idx - len(token) + 1
            span = (start_idx, end_idx + 1)
            found_tokens.append((token, span))
        return found_tokens

    def _build_trie(self):
        self._trie = self.build_trie(self.tokens)

    @staticmethod
    def build_trie(tokens: List[str]):
        trie = ahocorasick.Automaton()
        for idx, token in enumerate(tokens):
            trie.add_word(token, (idx, token))
        trie.make_automaton()
        return trie

    def reset(self):
        self.tokens = []


class LatticeBase(object):

    def __init__(self,
                 keep_unigram_node: bool = False,
                 directional: str = 'bidirectional'):
        self._graph = None
        self.keep_unigram_node = keep_unigram_node
        self.directional = directional

    def __repr__(self):
        if self._graph:
            nodes = list(self._graph.nodes)
            edges = list(self._graph.edges)
            return 'Lattice: V({})={}..., E({})={}..., directional={}'.format(
                len(nodes), nodes[:5], len(edges), edges[:5], self.directional)
        return ''

    def _get_nodes(self) -> List:
        return list(self._graph.nodes(data=True))

    def _get_edges(self,
                   return_tensor: bool = False) -> Union[torch.Tensor, List]:
        return self.get_edges(self._graph,
                              self.directional,
                              return_tensor=return_tensor)

    def _get_tokens(self) -> List:
        nodes = self._get_nodes()
        return [node[1]['token'] for node in nodes] if len(nodes) > 0 else None

    def _get_pyg_graph(self, group_node_attrs: List[str]):
        return self.get_pyg_graph(self._graph,
                                  group_node_attrs=group_node_attrs)

    @staticmethod
    def get_edges(graph: nx.DiGraph,
                  directional: str = 'bidirectional',
                  return_tensor: bool = True) -> Union[torch.Tensor, List]:
        _edges = list(graph.edges())
        if directional == 'forward':
            edges = _edges
        elif directional == 'backward':
            edges = [(edge[1], edge[0]) for edge in _edges]
        else:
            f_edges = _edges
            b_edges = [(edge[1], edge[0]) for edge in _edges]
            edges = f_edges + b_edges
        return LatticeBase.edges2tensor(edges) if return_tensor else edges

    @staticmethod
    def add_edges(graph, directional: str = 'forward') -> nx.DiGraph:
        allowed_directions = ['forward', 'backward', 'bidirectional']
        if directional not in allowed_directions:
            raise ValueError('directional: [{}]'.format(allowed_directions))

        nodes = graph.nodes()
        # add self-loop to avoid zero-edge lattice)
        graph.add_edge(0, 0)
        for i, u in enumerate(nodes):
            cur_node = nodes[u]
            for j, v in enumerate(nodes):
                if i != j:
                    next_node = nodes[v]
                    if cur_node['span'][1] == next_node['span'][0]:
                        if directional == 'forward':
                            graph.add_edge(u, v)
                        elif directional == 'backward':
                            graph.add_edge(v, u)
                        elif directional == 'bidirectional':
                            graph.add_edge(u, v)
                            graph.add_edge(v, u)
        return graph

    @staticmethod
    def add_special_edges(graph,
                          special_nodes: List,
                          directional: str = 'forward') -> nx.DiGraph:
        '''
        add edges from every nodes to special nodes
        '''
        allowed_directions = ['forward', 'backward', 'bidirectional']
        if directional not in allowed_directions:
            raise ValueError('directional: [{}]'.format(allowed_directions))

        nodes = graph.nodes()
        for i, u in enumerate(nodes):
            '''source node'''
            src_node = nodes[u]
            if (src_node['token'] not in special_nodes
                    and src_node['span'][1] - src_node['span'][0] != 0):
                for j, v in enumerate(nodes):
                    '''candidate target node'''
                    tgt_node = nodes[v]
                    if tgt_node['token'] in special_nodes:
                        '''add edges from source node to target node'''
                        if directional == 'forward':
                            graph.add_edge(u, v)
                        elif directional == 'backward':
                            graph.add_edge(v, u)
                        elif directional == 'bidirectional':
                            graph.add_edge(u, v)
                            graph.add_edge(v, u)
        return graph

    @staticmethod
    def edges2tensor(edges: List[Tuple(int, int)]) -> torch.Tensor:
        return torch.tensor(edges, dtype=torch.long).transpose(0, 1)

    @staticmethod
    def get_pyg_graph(graph, group_node_attrs: List[str]):
        return from_networkx(graph, group_node_attrs=group_node_attrs)


class BertLattice(LatticeBase):

    def __init__(self,
                 tokens: Union[str, List],
                 trie: Trie,
                 tokenizer,
                 vocab: Optional[Dict] = None,
                 pad_token: str = '[PAD]',
                 unk_token: str = '[UNK]',
                 bos_token: str = '[BOS]',
                 eos_token: str = '[EOS]',
                 dataset_token: Optional[str] = None,
                 max_token_length: Optional[int] = None,
                 node_comp_type: str = 'none',
                 rand_dropout: float = 0.0,
                 keep_unigram_node: bool = False,
                 directional: str = 'bidirectional'):
        super().__init__(keep_unigram_node, directional)
        self._build_graph_from_trie(tokens, trie, tokenizer, vocab, pad_token,
                                    unk_token, bos_token, eos_token,
                                    dataset_token, max_token_length,
                                    node_comp_type, rand_dropout,
                                    keep_unigram_node, directional)

    def _get_pyg_graph(self, group_node_attrs: List[str] = ['node_id']):
        return super()._get_pyg_graph(group_node_attrs=group_node_attrs)

    def _build_graph_from_trie(self, tokens: Union[str, List], trie: Trie,
                               tokenizer, vocab: Dict, pad_token: str,
                               unk_token: str, bos_token: str, eos_token: str,
                               dataset_token: str, max_token_length: int,
                               node_comp_type: str, rand_dropout: float,
                               keep_unigram_node: bool, directional: str):
        self._graph = self.build_graph_from_trie(
            tokens=tokens,
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
            directional=directional)

    @staticmethod
    def build_graph_from_trie(
        tokens: Union[str, List],
        trie: Trie,
        tokenizer,
        vocab: Dict,
        pad_token: str = '[PAD]',
        unk_token: str = '[UNK]',
        bos_token: str = '[BOS]',
        eos_token: str = '[EOS]',
        dataset_token: Optional[str] = None,
        max_token_length: Optional[int] = None,
        node_comp_type: str = 'none',
        rand_dropout: float = 0.0,
        keep_unigram_node: bool = False,
        directional: str = 'bidirectional',
    ) -> nx.DiGraph:
        if isinstance(tokens, list):
            tokens = ''.join(tokens)
        token_len = len(tokens)

        token_id_lookup_fn = tokenizer.convert_tokens_to_ids

        # nodes
        g = nx.DiGraph()
        g.add_node(0,
                   node_id=0,
                   span=(-1, -1),
                   token=pad_token,
                   token_id=[token_id_lookup_fn(pad_token)],
                   is_special_token=True)
        g.add_node(1,
                   node_id=1,
                   span=(0, 0),
                   token=bos_token,
                   token_id=[token_id_lookup_fn(bos_token)],
                   is_special_token=True)
        g.add_node(2,
                   node_id=2,
                   span=(token_len, token_len),
                   token=eos_token,
                   token_id=[token_id_lookup_fn(eos_token)],
                   is_special_token=True)

        node_idx = 3  # 0: pad, 1: bos, 2: eos
        if dataset_token is not None:
            g.add_node(3,
                       node_id=3,
                       span=(0, 0),
                       token=dataset_token,
                       token_id=[token_id_lookup_fn(dataset_token)],
                       is_special_token=True)
            node_idx = 4

        for (end_idx, (insert_order, token)) in trie.iter(tokens):
            start_idx = end_idx - len(token) + 1
            span = (start_idx, end_idx + 1)

            if max_token_length:
                if len(token) > max_token_length:
                    continue
            if random.random() < rand_dropout:
                if (len(token) != 1
                        or (len(token) == 1 and not keep_unigram_node)):
                    continue
            if not tokens[span[0]:span[1]] == token:
                raise AssertionError

            token_id = token_id_lookup_fn(token)
            token_id = [token_id]

            g.add_node(node_idx,
                       node_id=node_idx,
                       span=span,
                       token=token,
                       token_id=token_id,
                       is_special_token=False)
            node_idx += 1

        # edges
        g = LatticeBase.add_edges(g, directional=directional)

        return g

    @staticmethod
    def is_unk_token(token_id: int, tokenizer, unk_token='[UNK]') -> bool:
        return tokenizer.convert_ids_to_tokens(token_id) == unk_token
