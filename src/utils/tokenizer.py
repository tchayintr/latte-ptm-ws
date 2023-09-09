import json
import os
from transformers import BertTokenizer
from typing import Dict, Optional, List


def construct_bert_tokenizer(
        vocab: Dict[str, int],
        min_frequency: int = 1,
        clean_text: bool = False,
        handle_chinese_chars: bool = False,
        strip_accents: bool = False,
        lowercase: bool = False,
        max_length: int = 512,
        pad_token: str = '[PAD]',
        unk_token: str = '[UNK]',
        mask_token: str = '[MASK]',
        cls_token: str = '[CLS]',
        sep_token: str = '[SEP]',
        additional_special_tokens: Optional[List] = [
            '[BOS]',
            '[EOS]',
        ],
        save_path: str = 'models/tokenizers/bert_tokenizer_dev',
        return_tokenizer: bool = False) -> BertTokenizer:
    special_tokens = []
    if pad_token:
        special_tokens.append(pad_token)
    if unk_token:
        special_tokens.append(unk_token)
    if mask_token:
        special_tokens.append(mask_token)
    if cls_token:
        special_tokens.append(cls_token)
    if sep_token:
        special_tokens.append(sep_token)
    if additional_special_tokens is not None:
        special_tokens = special_tokens + additional_special_tokens
    '''save vocab and config'''
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    '''make vocab.txt'''
    save_bert_tokenizer_vocab(save_path,
                              vocab=vocab,
                              special_tokens=special_tokens)
    '''make config.json'''
    save_bert_tokenizer_config(save_path,
                               tokenizer_class='BertTokenizer',
                               vocab_size=len(vocab),
                               do_lower_case=lowercase,
                               pad_token=pad_token,
                               unk_token=unk_token,
                               mask_token=mask_token,
                               cls_token=cls_token,
                               sep_token=sep_token,
                               model_max_length=max_length,
                               max_len=max_length,
                               tokenize_chinese_chars=handle_chinese_chars)

    if return_tokenizer:
        return load_bert_tokenizer_from_path(
            save_path,
            do_lower_case=lowercase,
            pad_token=pad_token,
            unk_token=unk_token,
            mask_token=mask_token,
            cls_token=cls_token,
            sep_token=sep_token,
            tokenize_chinese_chars=handle_chinese_chars)


def load_bert_tokenizer_from_path(path: str,
                                  do_lower_case: bool,
                                  pad_token: str = '[PAD]',
                                  unk_token: str = '[UNK]',
                                  mask_token: str = '[MASK]',
                                  cls_token: str = '[CLS]',
                                  sep_token: str = '[SEP]',
                                  tokenize_chinese_chars: bool = False):
    vocab_file_path = os.path.join(path, 'vocab.txt')
    return BertTokenizer(vocab_file=vocab_file_path,
                         do_lower_case=do_lower_case,
                         pad_token=pad_token,
                         unk_token=unk_token,
                         mask_token=mask_token,
                         cls_token=cls_token,
                         sep_token=sep_token,
                         tokenize_chinese_chars=tokenize_chinese_chars)


def save_bert_tokenizer_vocab(path: str,
                              vocab: List,
                              special_tokens: Optional[List] = None):
    vocab_file_path = os.path.join(path, 'vocab.txt')
    with open(vocab_file_path, 'w') as f:
        if special_tokens is None:
            special_tokens = []

        for spt in special_tokens:
            print(spt, file=f)
        for token in vocab:
            if token not in special_tokens:
                print(token, file=f)


def save_bert_tokenizer_config(path: str,
                               tokenizer_class: str,
                               vocab_size: int,
                               do_lower_case: bool,
                               pad_token: str = '[PAD]',
                               unk_token: str = '[UNK]',
                               mask_token: str = '[MASK]',
                               cls_token: str = '[CLS]',
                               sep_token: str = '[SEP]',
                               model_max_length: bool = 512,
                               max_len: bool = 512,
                               tokenize_chinese_chars: bool = False):
    config_file_path = os.path.join(path, 'config.json')
    with open(config_file_path, 'w') as f:
        tokenizer_cfg = {
            'tokenize_chinese_chars': tokenize_chinese_chars,
            'vocab_size': vocab_size,
            'do_lower_case': do_lower_case,
            'pad_token': pad_token,
            'unk_token': unk_token,
            'mask_token': mask_token,
            'cls_token': cls_token,
            'sep_token': sep_token,
            'model_max_length': model_max_length,
            'max_len': max_len,
        }
        json.dump(tokenizer_cfg, f, indent=4)
