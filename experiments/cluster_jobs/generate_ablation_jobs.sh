#!/bin/bash
# ============================================================
# generate_ablation_jobs.sh
# ============================================================
# Generates 8 individual LSF scripts (one per ablation mode)
# and a comparison script to run after all jobs complete.
#
# Usage:
#   bash generate_ablation_jobs.sh
#   # Then submit:
#   for f in ablation_*.lsf; do bsub < $f; done
#   # After all complete:
#   python compare_ablation.py --results_dir $OUT_DIR

# ---- EDIT THESE PATHS ----
DATA_CSV="${SPECTRA_ROOT}/app/ImmunoStruct/data/training_rosetta.csv"
PROJECT_DIR="${SPECTRA_ROOT}/project/TCRpMHC"
OUT_DIR="${PROJECT_DIR}/outputs/ablation"
ESM_CKPT="${SPECTRA_ROOT}/data/models/facebook/esm2_t6_8M_UR50D"
CONDA_ENV="immunostruct"
# --------------------------

MODES=("A" "B" "C" "D" "E" "F" "G" "H")
NAMES=("concat_cls" "4chain_pool" "4chain_pool_rosetta" "4chain_crossattn" "4chain_crossattn_rosetta" "concat_cls_rosetta" "2chain_pool" "2chain_pool_rosetta")

# Estimated wall times per mode (hours)
WALL=("6:00" "12:00" "12:00" "16:00" "16:00" "6:00" "8:00" "8:00")

# Memory per mode (GB) — 4-chain modes need more for 4 ESM passes
MEM=("48" "64" "64" "64" "64" "48" "48" "48")

for i in "${!MODES[@]}"; do
    MODE=${MODES[$i]}
    NAME=${NAMES[$i]}
    W=${WALL[$i]}
    M=${MEM[$i]}
    LSF_FILE="ablation_${MODE}.lsf"

    cat > "$LSF_FILE" << ENDOFLSF
#!/bin/bash
#BSUB -J abl_${MODE}
#BSUB -q egpu
#BSUB -n 8
#BSUB -M ${M}
#BSUB -R "rusage[mem=${M}]"
#BSUB -gpu "num=1:gmem=16"
#BSUB -W ${W}
#BSUB -o ${OUT_DIR}/mode_${MODE}/job_%J.out
#BSUB -e ${OUT_DIR}/mode_${MODE}/job_%J.err
#BSUB -cwd ${PROJECT_DIR}

# ============================================================
# Ablation Mode ${MODE}: ${NAME}
# ============================================================

source activate ${CONDA_ENV}

mkdir -p ${OUT_DIR}/mode_${MODE}

echo "======================================"
echo "  Mode ${MODE}: ${NAME}"
echo "  Start: \$(date)"
echo "  GPU: \$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "======================================"

cd "$HOME/app/ImmunoStruct/esm_rosetta"
python run_ablation.py \\
    --modes ${MODE} \\
    --data_csv ${DATA_CSV} \\
    --esm_checkpoint ${ESM_CKPT} \\
    --d_model 128 \\
    --n_cross_heads 4 \\
    --d_rosetta 64 \\
    --d_fused 256 \\
    --dropout 0.2 \\
    --batch_size 16 \\
    --epochs 30 \\
    --lr 3e-4 \\
    --esm_lr 1e-5 \\
    --patience 7 \\
    --crystal_weight 5.0 \\
    --unfreeze_b 10 \\
    --unfreeze_c 20 \\
    --val_split 0.15 \\
    --seed 42 \\
    --num_workers 4 \\
    --out_dir ${OUT_DIR} \\
    --progress_bar

echo "======================================"
echo "  Mode ${MODE} complete: \$(date)"
echo "  Result: ${OUT_DIR}/mode_${MODE}/result.json"
echo "======================================"
ENDOFLSF

    echo "Generated: ${LSF_FILE}"
done

echo ""
echo "Submit all jobs:"
echo "  for f in ablation_*.lsf; do bsub < \$f; done"
echo ""
echo "After all complete, run comparison:"
echo "  python compare_ablation.py --results_dir ${OUT_DIR}"