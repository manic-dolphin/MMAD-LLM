#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
OUTPUT=$1
ZERO_STAGE=$2
if [ "$OUTPUT" == "" ]; then
    OUTPUT=./output_step1_llama2_7b_lora_gnn_test_2
fi
if [ "$ZERO_STAGE" == "" ]; then
    ZERO_STAGE=3
fi
mkdir -p $OUTPUT

deepspeed deepspeed_ft_test.py \
   --data_path './data/chem_data/orderly_train_with_graph_test' \
   --data_split 1,0,0 \
   --model_name_or_path '/data/yanyuliang/Code/got/hf_models/llama2/llama2-7b-chat' \
   --per_device_train_batch_size 8 \
   --per_device_eval_batch_size 4 \
   --max_seq_len 128 \
   --learning_rate 9.65e-6 \
   --weight_decay 0. \
   --num_train_epochs 3  \
   --gradient_accumulation_steps 4 \
   --lr_scheduler_type cosine \
   --num_warmup_steps 0 \
   --seed 1234 \
   --gradient_checkpointing \
   --zero_stage $ZERO_STAGE \
   --deepspeed \
   --lora_dim 128 \
   --lora_module_name "layers." \
   --output_dir $OUTPUT \
   &> $OUTPUT/training.log