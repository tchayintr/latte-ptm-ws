"""Regression test: the lattice path (BiGAT and word nodes) must receive gradients
during multi-criteria pre-training.

Runs one real batch of the bundled Thai samples through LatteBertPretrainer.
Set LATTE_TEST_PTM to a local BERT directory to avoid downloading
bert-base-multilingual-cased.

    python tests/test_gradient_flow.py        (or: python -m pytest tests/)
"""
import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))


def test_lattice_path_receives_gradients():
    os.chdir(ROOT)
    os.cpu_count = lambda: 4  # one dataloader worker
    import pretrain
    from pretrainers.latte_bert_pretrainer import LatteBertPretrainer

    ptm = os.environ.get('LATTE_TEST_PTM', 'bert-base-multilingual-cased')
    out = tempfile.mkdtemp()
    sys.argv = [
        'pretrain.py', '--run', 'latte', '--data-dir', 'data/samples/th',
        '--save-dir', out, '--pretrained-save-path', out,
        '--pretrained-model', ptm, '--batch-size', '4', '--bert-mode', 'sum',
        '--optimized-decay', '--scheduler', '--lang', 'th',
        '--normalize-unicode', '--max-token-length', '12',
        '--train-split-ratio', '0.9', '--gnn-type', 'gat',
        '--criterion-type', 'crf-nll', '--metric-type', 'word-bin',
        '--attn-comp-type', 'wavg', '--node-comp-type', 'none',
        '--ext-dic-file', 'data/dict/samples/hse-thai-lexitron.vocab.sample.sl',
        '--unc-token-ratio', '0.1', '--graph-dropout', '0.2',
        '--include-dataset-token', '--include-unc-token', '--seed', '112'
    ]
    args = pretrain.get_args()
    pretrain.set_seeds(args.seed)
    model = LatteBertPretrainer(args)
    model.train()

    xs, ys = next(iter(model.train_dataloader()))
    loss = model._compute_active_loss(xs, ys, model.forward(xs))
    loss.backward()

    no_grad = [
        n for n, p in model.gnn.named_parameters()
        if p.grad is None or not bool(p.grad.abs().sum() > 0)
    ]
    assert not no_grad, f'GNN parameters without gradient: {no_grad}'

    word_ids = [
        d.token_id.view(-1)[[i for i, (s, e) in enumerate(d.span) if e - s > 1]]
        for d in xs['lattice'].to_data_list()
    ]
    word_ids = __import__('torch').cat(word_ids).unique()
    grad = model.bert.embeddings.word_embeddings.weight.grad
    assert grad is not None and bool(
        (grad[word_ids].abs().sum(-1) > 0).all()), \
        'word-node (lexicon token) embeddings received no gradient'


if __name__ == '__main__':
    test_lattice_path_receives_gradients()
    print('ok: BiGAT and word nodes receive gradients')
