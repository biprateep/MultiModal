#!/bin/bash
set -euo pipefail
cd /pscratch/sd/p/pzehao/MultiModal
PYTHON_BIN="/global/cfs/cdirs/desi/users/pzehao/envs/peng/bin/python"
LOG_DIR="$1"
COMMON=(
  --mode interleave
  --end 4737442
  --train-size 4642694
  --batch-size 32
  --num-workers 7
  --prefetch-factor 4
  --distributed-shard-mode contiguous
  --num-nodes 4
  --devices 4
  --warmup-steps 2
  --measure-steps 6
  --ckpt-path /pscratch/sd/p/pzehao/DESIMAE/ProductionCheckpointsFinal/epoch=015-val_loss=-1.2884.ckpt
)
run_case () {
  NAME="$1"
  shift
  echo "===== START ${NAME} ====="
  set +e
  timeout 8m srun --nodes=4 --ntasks-per-node=4 --cpus-per-task=32 \
    "$PYTHON_BIN" train/DDPModeBench.py "${COMMON[@]}" "$@" \
    > "$LOG_DIR/${NAME}.out" 2> "$LOG_DIR/${NAME}.err"
  RC=$?
  set -e
  echo "RC ${NAME}=${RC}"
  grep -n "DDP_BENCH_START\|THROUGHPUT_RESULT" "$LOG_DIR/${NAME}.out" || true
  echo "===== END ${NAME} ====="
}

# Sweep train_subset_size (block size fixed at 100)
run_case S0_B100     --train-subset-size 0      --train-block-size 100
run_case S25K_B100   --train-subset-size 25000  --train-cycle-epoch 0 --train-block-size 100
run_case S50K_B100   --train-subset-size 50000  --train-cycle-epoch 0 --train-block-size 100
run_case S100K_B100  --train-subset-size 100000 --train-cycle-epoch 0 --train-block-size 100

# Sweep train_block_size (subset size fixed at 50000)
run_case S50K_B64    --train-subset-size 50000  --train-cycle-epoch 0 --train-block-size 64
run_case S50K_B100_R --train-subset-size 50000  --train-cycle-epoch 0 --train-block-size 100
run_case S50K_B128   --train-subset-size 50000  --train-cycle-epoch 0 --train-block-size 128
run_case S50K_B256   --train-subset-size 50000  --train-cycle-epoch 0 --train-block-size 256

echo "SWEEP_LOG_DIR=$LOG_DIR"
