import argparse
import datetime
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (ModelCheckpoint, LearningRateMonitor)
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.loggers import TensorBoardLogger
from lightning_lite.utilities.seed import seed_everything
import random
import torch
from transformers import set_seed

from pretrainers.bert_pretrainer import BertPretrainer
from pretrainers.latte_bert_pretrainer import LatteBertPretrainer


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-gpus', type=int, default=1)
    parser.add_argument('--ddp-timeout', type=int, default=7200)
    parser.add_argument('--ckpt-path',
                        help='Specify a checkpoint path to resume training')
    parser.add_argument('--normalize-unicode', action='store_true')
    parser.add_argument('--lang', choices=['zh', 'ja', 'th'], default='zh')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--criterion-type',
                        choices=['crf-nll'],
                        default='crf-nll')
    parser.add_argument('--metric-type',
                        choices=['word-cws', 'word-bin'],
                        default='word-bin')
    parser.add_argument('--use-binary', action='store_true')
    parser.add_argument('--run', choices=['latte'], default='latte')
    parser = pl.Trainer.add_argparse_args(parser)
    parser = BertPretrainer.add_specific_args(parser)
    parser = LatteBertPretrainer.add_specific_args(parser)
    args = parser.parse_args()
    return args


def set_seeds(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    g = torch.Generator()
    g.manual_seed(seed)
    set_seed(seed)
    seed_everything(seed)


def load_model_from_checkpoint(model_type, checkpoint_path):
    if model_type == 'latte':
        return LatteBertPretrainer.load_from_checkpoint(checkpoint_path)
    else:
        raise NotImplementedError


def run(args):
    if args.seed is not None:
        set_seeds(args.seed)

    if args.run == 'latte':
        print('Model: LatteBertPretrainer')
        model = LatteBertPretrainer(args)
    else:
        raise NotImplementedError

    if args.ckpt_path:
        print('Resume: {}'.format(args.ckpt_path))

    checkpoint_callback = ModelCheckpoint(
        filename='{epoch}-{valid_loss:.4f}-{valid_f1:.4f}',
        monitor='valid_f1',
        mode='max',
        save_top_k=1,
        verbose=True)
    lr_callback = LearningRateMonitor(logging_interval='epoch')
    logger = TensorBoardLogger(save_dir=args.save_dir,
                               name=args.model_name,
                               version=args.model_version,
                               default_hp_metric=False)
    trainer = pl.Trainer.from_argparse_args(
        args,
        accelerator='gpu',
        devices=args.num_gpus,
        default_root_dir=args.save_dir,
        auto_lr_find=False,
        callbacks=[checkpoint_callback, lr_callback],
        strategy=DDPStrategy(find_unused_parameters=True,
                             timeout=datetime.timedelta(
                                 seconds=args.ddp_timeout))
        if args.num_gpus > 1 else None,
        replace_sampler_ddp=(False if args.num_gpus > 1 else True),
        logger=logger)
    trainer.fit(model, ckpt_path=args.ckpt_path)

    best_model_path = checkpoint_callback.best_model_path
    print('Best model: {}'.format(best_model_path))

    # save the best pretrained model
    if args.num_gpus > 1:
        torch.distributed.barrier()
    print('Load best model from its checkpoint: {}'.format(best_model_path))
    best_model = load_model_from_checkpoint(args.run, best_model_path)
    best_model.save_pretrained_bert(args.pretrained_save_path)
    print('Save pretrained best model: {}'.format(args.pretrained_save_path))


if __name__ == '__main__':
    args = get_args()
    run(args)
