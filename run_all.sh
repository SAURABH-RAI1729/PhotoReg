#!/bin/bash

# ============================================================
# run_all.sh
#
# Runs the full pipeline (train -> evaluate -> infer -> uncertainty)
# for all three architectures: MLP, CNN, DeepSets.
#
# Usage:
#   chmod +x run_all.sh
#   ./run_all.sh
# ============================================================

set -e

EPOCHS=30
CONFIG="configs/config.yaml"
ARCHITECTURES=("mlp" "cnn" "deepsets")

run_pipeline() {

    local ARCH=$1

    echo ""
    echo "============================================================"
    echo "  Architecture: ${ARCH^^}"
    echo "============================================================"

    cp $CONFIG ${CONFIG}.bak

    # Patch architecture
    sed -i "s/^\(\s*architecture:\s*\).*/\1\"${ARCH}\"/" $CONFIG

    # Patch epochs
    sed -i "s/^\(\s*epochs:\s*\).*/\1${EPOCHS}/" $CONFIG

    # Patch plots dir -- match the line after 'plots:' specifically
    # by using a two-line address range in sed
    sed -i "/^plots:/{n; s|^\(\s*dir:\s*\).*|\1\"plots/${ARCH}\"|;}" $CONFIG

    echo ""
    echo "--- Training ---"
    python train.py --config $CONFIG

    echo ""
    echo "--- Evaluation ---"
    python evaluate.py --config $CONFIG

    echo ""
    echo "--- Inference Benchmark ---"
    python infer.py --config $CONFIG

    echo ""
    echo "--- Uncertainty Suite ---"
    python uncertainty_runner.py --config $CONFIG

    mv ${CONFIG}.bak $CONFIG

    echo ""
    echo "  Done: ${ARCH^^}. Results saved to plots/${ARCH}/"
}

echo ""
echo "Starting full pipeline for all architectures"
echo "Epochs per model : ${EPOCHS}"
echo "Config           : ${CONFIG}"

for ARCH in "${ARCHITECTURES[@]}"; do
    run_pipeline $ARCH
done

echo ""
echo "============================================================"
echo "  All done."
echo ""
echo "  Results:"
for ARCH in "${ARCHITECTURES[@]}"; do
    echo "    plots/${ARCH}/"
done
echo "  Checkpoints: checkpoints/"
echo "============================================================"
