#!/usr/bin/env bash
set -euo pipefail

# Phase-1 smoke: pseudo-quant only (no real_quant / CUDA deploy)

cd "$(dirname "$0")/.."

MODEL="llava_onevision"
MODEL_ARGS="pretrained=your/model-name"

DATA_PATH="${DATA_PATH:-}"
IMAGE_FOLDER="${IMAGE_FOLDER:-}"

N_SAMPLES=8
W_BIT=4
W_GROUP=128
OUT_DIR="quick_smoke_${MODEL}"
SCALE_PATH="${OUT_DIR}/prism_w4.pt"
RESULTS_MD="${OUT_DIR}/smoke_results.md"

mkdir -p "${OUT_DIR}"

if [[ -z "${DATA_PATH}" || -z "${IMAGE_FOLDER}" ]]; then
  echo "Set DATA_PATH and IMAGE_FOLDER to COCO calib paths."
  exit 1
fi

echo "[1/2] Pseudo-quant smoke..."
python main_quant.py \
  --model "${MODEL}" \
  --model_args "${MODEL_ARGS}" \
  --batch_size 1 \
  --calib_data coco \
  --n_samples "${N_SAMPLES}" \
  --data_path "${DATA_PATH}" \
  --image_folder "${IMAGE_FOLDER}" \
  --run_process \
  --pseudo_quant \
  --w_bit "${W_BIT}" \
  --w_group "${W_GROUP}" \
  --asd_mixed_precision \
  --asd_theta1 0.8 \
  --asd_theta2 0.2 \
  --asd_high_precision_ratio 0.01 \
  --asd_low_w_bit 4 \
  --scale_path "${SCALE_PATH}"

echo "[2/2] Eval smoke..."
python main_eval.py \
  --model "${MODEL}" \
  --model_args "${MODEL_ARGS}" \
  --batch_size 1 \
  --tasks "mmmu_val" \
  --limit 2 \
  --scale_path "${SCALE_PATH}" \
  --pseudo_quant \
  --output_path "${OUT_DIR}" \
  --results_md "${RESULTS_MD}"

echo "Done: ${OUT_DIR}"
