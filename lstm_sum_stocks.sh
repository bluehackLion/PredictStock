#! /bin/bash
source activate demand_prediction
export OUTPUT_PATH='pwd'

python ./lstm_sum_stocks.py \
    --input_data_column_cnt=8 \
    --output_data_column_cnt=1 \
    --seq_length=10 \
    --rnn_cell_hidden_dim=30 \
    --forget_bias=1.0 \
    --num_stacked_layers=1 \
    --keep_prob=0.7 \
    --epoch_num=8000 \
    --learning_rate=0.001 \
    --output_path=$OUTPUT_PATH
