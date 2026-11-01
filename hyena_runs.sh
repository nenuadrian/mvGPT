#!/usr/bin/env bash
# Episode 29's small models, one after another on one GPU, half an hour each: 6 blocks of 384 channels with attention,
# with Hyena, and with Hyena but attention in block 3, on windows of 1,024 and of 4,096 tokens. Each step is 65,536
# tokens, in two batches.
#   CUDA_VISIBLE_DEVICES=1 ./hyena_runs.sh
set -euo pipefail
mkdir -p logs
for block in 1024 4096; do
  for kind in attention hyena hybrid; do
    case $kind in
      attention) flags=() ;;
      hyena) flags=(--hyena) ;;
      hybrid) flags=(--hyena --attention-layers 3) ;;
    esac
    torchrun --standalone --nproc_per_node 1 pretrain_ddp.py 30 --web --compile --layers 6 --heads 6 --embd 384 \
      --block $block --batch $((32768 / block)) --accum 2 "${flags[@]}" --name small-$kind-$block \
      > logs/small-$kind-$block.log 2>&1
  done
done
