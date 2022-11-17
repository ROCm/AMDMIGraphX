#!/bin/bash
MODEL=$1
LOG="ck_bbc.log"
TUNING_DB="ck_bbc.json"

rm $LOG
touch $LOG
for N in 1 16 32 64
do
    MIGRAPHX_LOG_CK_GEMM=1 ./bin/driver run $MODEL -g --fill1 input_ids --input-dim @input_ids $N 384  | grep 'ck_gemm.*: \[{' | sort -u >> $LOG
done

python3 ../tools/tune_ck.py -n 16 -l $LOG -o $TUNING_DB
export MIGRAPHX_CK_TUNING=$TUNING_DB