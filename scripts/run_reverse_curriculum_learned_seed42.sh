#!/bin/bash
set -e

export PYTHONPATH=.

LOG_DIR="outputs/logs/mitigation"
mkdir -p "$LOG_DIR"

CONFIGS=(
  "configs/mitigation/reverse_curriculum_learned_seed42_stage1.yaml"
  "configs/mitigation/reverse_curriculum_learned_seed42_stage2.yaml"
  "configs/mitigation/reverse_curriculum_learned_seed42_stage3.yaml"
  "configs/mitigation/reverse_curriculum_learned_seed42_stage4.yaml"
)

echo "Starting reverse learned curriculum experiment"

for CONFIG in "${CONFIGS[@]}"
do
  NAME=$(basename "$CONFIG" .yaml)

  echo "========================================"
  echo "Generating dataset for $NAME"
  echo "========================================"

  python -m src.tasks.generate_dataset \
    --config "$CONFIG" \
    2>&1 | tee "$LOG_DIR/${NAME}_generate.log"

  echo "========================================"
  echo "Training $NAME"
  echo "========================================"

  python -m src.training.train \
    --config "$CONFIG" \
    2>&1 | tee "$LOG_DIR/${NAME}_train.log"
done

echo "========================================"
echo "Evaluating final stage"
echo "========================================"

python -m src.training.evaluate \
  --config "configs/mitigation/reverse_curriculum_learned_seed42_stage4.yaml" \
  2>&1 | tee "$LOG_DIR/reverse_curriculum_learned_seed42_stage4_evaluate.log"

echo "Done."
echo "Final result:"
cat outputs/results/mitigation_results/reverse_train128_curriculum_learned_seed42_stage4_results.csv