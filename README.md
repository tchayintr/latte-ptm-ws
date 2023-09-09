<div align="center">

<h3>Multi-criteria Word Segmentation Pre-training with LATTE</h3>

_latte-ptm-ws_

______________________________________________________________________

[![Python Version](https://img.shields.io/badge/python-3.9-blue)](https://github.com/tchayintr/latte-ptm-ws)
[![PyTorch Version](https://img.shields.io/badge/torch-2.0.1-blue)](https://pytorch.org/get-started/locally/)
[![Lightning Version](https://img.shields.io/badge/lightning-1.9.5-blue)](https://pypi.org/project/pytorch-lightning/1.9.5/)
[![PyG Version](https://img.shields.io/badge/pyg-2.3.1-blue)](https://pypi.org/project/torch-geometric/2.3.1/)
[![AllenNLP Light Version](https://img.shields.io/badge/allennlp--light-1.0.0*-blue)](https://github.com/tchayintr/allennlp-light/tree/crf-allowed-transitions-patch)
![CUDA Version](https://img.shields.io/badge/CUDA-11.8-green)
[![Apache License](https://img.shields.io/badge/License-Apache%202.0-blue)](https://github.com/tchayintr/latte-ptm-ws/blob/main/LICENSE)

</div>

### LATTE: Lattice ATTentive Encoding for Character-based Word Segmentation
- https://github.com/tchayintr/latte-ws

### Architecture
- Character-based word segmentation
- Multi-granularity Lattice (character-word)
    - Encoded with Bidirectional-GAT
- Pre-training and Fine-tuning methods
- BMES tagging scheme
    - B: beginning, M: middle, E: end, and S: single

### Segmentation Performance (including char-bin-f1, word-f1, oov-recall)
- CTB6 (zh): 
    - word-f1: 98.1
    - oov-recall: 90.6
- BCCWJ (ja): 
    - word-f1: 99.4
    - oov-recall: 92.1
- BEST2010 (th): 
    - char-bin-f1: 99.1
    - word-f1: 97.7

### Datasets (test sets were excluded)
- Seven Chinese datasets (converted into simplified Chinese)
    - CTB6 (main)
    - SIGHAN2005 (AS, CITYU, MSRA, PKU)
    - SIGHAN2008 (SXU)
    - CNC
- Five Thai datasets
    - BEST2010 (main)
    - LST20
    - TNHC
    - VISTEC
    - WS160
- Three Japanese datasets
    - BCCWJ (main)
    - UD Japanese treebank
    - Kyoto University Text Corpus

#### Dataset Notes
- Place datasets in the same directory.
	- For example, `data/zh/ctb6.train.sl`, `data/zh/as.train.sl`, etc.
- Format each dataset in `sl` (word-segmented sentence line).
	- In this format, each line contains a word-segmented sentence, with words separated by white spaces.

### Pre-trained Models can be found at
- zh: https://huggingface.co/yacht/latte-mc-bert-base-chinese-ws
- ja: https://huggingface.co/yacht/latte-mc-bert-base-japanese-ws
- th: https://huggingface.co/yacht/latte-mc-bert-base-thai-ws

### Saved Model Directories
- `model/`
    - PyTorch model files
- `pretrained/`
    - Pre-trained model files
    - Ready to be loaded by `transformers` library

### Requirements
- pip
    - requirements.txt
    - `pip install -r requirements.txt`
- conda
    - environment.yml
    - `conda env create -f environment.yml`

### Usage
- See `scripts/` for examples

### Citation
- Published in Journal of Natural Language Processing
    - https://www.jstage.jst.go.jp/article/jnlp/30/2/30_456/_article/-char/ja
