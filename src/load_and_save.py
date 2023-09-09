from pytorch_lightning.utilities.seed import seed_everything

import pretrain
from pretrainers.latte_bert_pretrainer import LatteBertPretrainer


def load_model_from_checkpoint(checkpoint_path):
    return LatteBertPretrainer.load_from_checkpoint(checkpoint_path)


def run(args):
    seed_everything(args.seed)
    print('Load model from its checkpoint: {}'.format(args.ckpt_path))
    best_model = load_model_from_checkpoint(args.run, args.ckpt_path)
    best_model.save_pretrained_bert(args.pretrained_save_path)
    print('Save pretrained best model: {}'.format(args.pretrained_save_path))


if __name__ == '__main__':
    args = pretrain.get_args()
    run(args)
