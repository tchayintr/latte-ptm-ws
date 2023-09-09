set -e

NUM_GPUS=1
DATA_DIR=data/ja
EXT_DIC_FILE=data/dict/unidic_3_1-ipadic.vocab.sl
PRETRAINED_MODEL=data/ptm/bert-base-japanese-char-v2
# PRETRAINED_MODEL=cl-tohoku/bert-base-japanese-char-v2
SAVE_DIR=models/ja
MODEL_NAME=latte-mc-bert-base-japanese-ws
MODEL_VERION=0
BATCH_SIZE=32
MAX_EPOCHS=20
ACC_GRAD_BATCH=4
GRADIENT_CLIP_VAL=5.0
BERT_MODE=sum
LR=1e-3
BERT_LR=2e-5
GNN_LR=1e-3
LR_DECAY_RATE=0.90
LANG=ja
MAX_TOKEN_LEN=4
GNN_TYPE=gat
ATTN_COMP_TYPE=wavg
METRIC_TYPE=word-bin
NODE_COMP_TYPE=none
TRAIN_RATIO=0.9
UNC_TOKEN_RATIO=0.1
DROPOUT=0.2
GRAPH_DROPOUT=0.2
ATTN_DROPOUT=0.2
CRITERION_TYPE=crf-nll
METRIC_TYPE=word-bin
RUN_MODE=latte
PRETRAINED_SAVE_PATH=pretrained/ja/$MODEL_NAME
SEED=112

python3 src/pretrain.py \
    --run $RUN_MODE \
    --data-dir $DATA_DIR \
    --save-dir $SAVE_DIR \
    --pretrained-model $PRETRAINED_MODEL \
    --model-name $MODEL_NAME \
    --model-version $MODEL_VERION \
    --max-epochs $MAX_EPOCHS \
    --batch-size $BATCH_SIZE \
    --accumulate-grad-batches $ACC_GRAD_BATCH \
    --gradient-clip-val $GRADIENT_CLIP_VAL \
    --bert-mode $BERT_MODE \
    --lr $LR \
    --bert-lr $BERT_LR \
    --gnn-lr $GNN_LR \
    --lr-decay-rate $LR_DECAY_RATE \
    --optimized-decay \
    --scheduler \
    --lang $LANG \
    --normalize-unicode \
    --max-token-length $MAX_TOKEN_LEN \
    --train-split-ratio $TRAIN_RATIO \
    --shuffle-data \
    --gnn-type $GNN_TYPE \
    --criterion-type $CRITERION_TYPE \
    --metric-type $METRIC_TYPE \
    --attn-comp-type $ATTN_COMP_TYPE \
    --node-comp-type $NODE_COMP_TYPE \
    --ext-dic-file $EXT_DIC_FILE \
    --pretrained-save-path $PRETRAINED_SAVE_PATH \
    --unc-token-ratio $UNC_TOKEN_RATIO \
    --dropout $DROPOUT \
    --graph-dropout $GRAPH_DROPOUT \
    --attn-dropout $ATTN_DROPOUT \
    --include-dataset-token \
    --include-unc-token \
    --use-binary \
    --seed $SEED \
    --generate-unigram-node \
    # --num-gpus $NUM_GPUS \
    #
